-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Praxis Contributors

local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local timeout_ms = tonumber(ARGV[1])
local max_keys = tonumber(ARGV[2])
local max_active = tonumber(ARGV[3])
local estimate = tonumber(ARGV[4])
local budget_count = tonumber(ARGV[5])
local settled = KEYS[2]
local active = KEYS[3]

local active_total = tonumber(redis.call('GET', KEYS[5]) or '0')
local rule_active_total = tonumber(redis.call('GET', KEYS[8]) or '0')
local function reported_remaining()
  return math.min(9007199254740991, tonumber(redis.call('GET', KEYS[12]) or '0'))
end

local function update_remaining(value)
  if redis.call('ZSCORE', KEYS[4], KEYS[1]) == false then return end
  local previous = tonumber(redis.call('HGET', KEYS[11], KEYS[1]) or '0')
  redis.call('INCRBY', KEYS[12], value - previous)
  redis.call('HSET', KEYS[11], KEYS[1], value)
end

-- Expired entries are drained a batch at a time so one admission never
-- blocks the server on a large backlog; later admissions finish the job.
local SWEEP_BATCH = 128

local expired_rule = redis.call('ZRANGE', KEYS[9], '-inf', now_ms, 'BYSCORE', 'LIMIT', 0, SWEEP_BATCH)
for i = 1, #expired_rule do
  local member = expired_rule[i]
  local split = string.find(member, '|')
  if split then
    local physical = string.sub(member, 1, split - 1)
    local reservation = string.sub(member, split + 1)
    local active_key = physical .. ':active'
    local value = redis.call('HGET', active_key, reservation)
    if value then
      local value_split = string.find(value, '|')
      local amount = tonumber(string.sub(value, 1, value_split - 1))
      local reserved_at = tonumber(string.sub(value, value_split + 1))
      redis.call('ZADD', physical .. ':settled', reserved_at, 'expired:' .. reservation .. ':' .. amount)
      redis.call('HDEL', active_key, reservation)
      active_total = math.max(0, active_total - 1)
    end
    rule_active_total = math.max(0, rule_active_total - 1)
  end
  redis.call('ZREM', KEYS[7], member)
  redis.call('ZREM', KEYS[9], member)
end
redis.call('SET', KEYS[8], rule_active_total)

local expired_global = redis.call('ZRANGE', KEYS[7], '-inf', now_ms, 'BYSCORE', 'LIMIT', 0, SWEEP_BATCH)
for i = 1, #expired_global do
  local member = expired_global[i]
  local split = string.find(member, '|')
  if split then
    local physical = string.sub(member, 1, split - 1)
    local reservation = string.sub(member, split + 1)
    local active_key = physical .. ':active'
    local value = redis.call('HGET', active_key, reservation)
    if value then
      local value_split = string.find(value, '|')
      local amount = tonumber(string.sub(value, 1, value_split - 1))
      local reserved_at = tonumber(string.sub(value, value_split + 1))
      redis.call('ZADD', physical .. ':settled', reserved_at, 'expired:' .. reservation .. ':' .. amount)
      redis.call('HDEL', active_key, reservation)
      active_total = math.max(0, active_total - 1)
    end
  end
  redis.call('ZREM', KEYS[7], member)
end
redis.call('SET', KEYS[5], active_total)

local expired_keys = redis.call('ZRANGE', KEYS[10], '-inf', now_ms, 'BYSCORE', 'LIMIT', 0, SWEEP_BATCH)
for i = 1, #expired_keys do
  local previous = tonumber(redis.call('HGET', KEYS[11], expired_keys[i]) or '0')
  redis.call('INCRBY', KEYS[12], -previous)
  redis.call('HDEL', KEYS[11], expired_keys[i])
  redis.call('ZREM', KEYS[10], expired_keys[i])
end

local max_window = 0
for i = 1, budget_count do
  local window = tonumber(ARGV[5 + (i * 2) - 1])
  if window > max_window then max_window = window end
  redis.call('ZREMRANGEBYSCORE', settled, '-inf', now_ms - window)
end
local telemetry_ttl = math.max(max_window + timeout_ms, 1000)
local function refresh_rule_telemetry_ttl()
  for i = 8, 12 do redis.call('PEXPIRE', KEYS[i], telemetry_ttl) end
end

local expired = {}
local active_values = redis.call('HGETALL', active)
for i = 1, #active_values, 2 do
  local id = active_values[i]
  local value = active_values[i + 1]
  local sep = string.find(value, '|')
  local reserved_at = tonumber(string.sub(value, sep + 1))
  if now_ms - reserved_at >= timeout_ms then
    local amount = tonumber(string.sub(value, 1, sep - 1))
    redis.call('ZADD', settled, reserved_at, 'expired:' .. id .. ':' .. amount)
    redis.call('HDEL', active, id)
    active_total = math.max(0, active_total - 1)
  end
end
redis.call('SET', KEYS[5], active_total)

redis.call('ZREMRANGEBYSCORE', KEYS[4], '-inf', now_ms)
local key_exists = redis.call('ZSCORE', KEYS[4], KEYS[1]) ~= false
if not key_exists and redis.call('ZCARD', KEYS[4]) >= max_keys then
  refresh_rule_telemetry_ttl()
  return {0, max_window, math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[10])}
end
if active_total >= max_active then
  refresh_rule_telemetry_ttl()
  return {0, max_window, math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[10])}
end

local key_remaining = nil
local max_usage = 0
for i = 1, budget_count do
  local window = tonumber(ARGV[5 + (i * 2) - 1])
  local capacity = tonumber(ARGV[5 + (i * 2)])
  local settled_sum = 0
  local entries = redis.call('ZRANGE', settled, now_ms - window, '+inf', 'BYSCORE', 'WITHSCORES')
  for j = 1, #entries, 2 do
    local member = entries[j]
    local amount = string.match(member, ':(%d+)$')
    if amount then settled_sum = settled_sum + tonumber(amount) end
  end
  local active_values = redis.call('HGETALL', active)
  local active_sum = 0
  for j = 1, #active_values, 2 do
    local sep = string.find(active_values[j + 1], '|')
    active_sum = active_sum + tonumber(string.sub(active_values[j + 1], 1, sep - 1))
  end
  local available = math.max(0, capacity - settled_sum - active_sum)
  if key_remaining == nil or available < key_remaining then key_remaining = available end
  local total_usage = settled_sum + active_sum + estimate
  if total_usage > max_usage then max_usage = total_usage end
  if total_usage > capacity then
    update_remaining(key_remaining)
    refresh_rule_telemetry_ttl()
    return {0, max_window, math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[10])}
  end
end

local id = redis.call('INCR', KEYS[6])
redis.call('HSET', active, id, estimate .. '|' .. now_ms)
redis.call('INCR', KEYS[5])
rule_active_total = rule_active_total + 1
redis.call('SET', KEYS[8], rule_active_total)
redis.call('ZADD', KEYS[7], now_ms + timeout_ms, KEYS[1] .. '|' .. id)
redis.call('ZADD', KEYS[9], now_ms + timeout_ms, KEYS[1] .. '|' .. id)
local ttl = math.max(max_window + timeout_ms, 1000)
redis.call('ZADD', KEYS[4], now_ms + ttl, KEYS[1])
redis.call('ZADD', KEYS[10], now_ms + ttl, KEYS[1])
redis.call('PEXPIRE', settled, ttl)
redis.call('PEXPIRE', active, ttl)
redis.call('PEXPIRE', KEYS[1], ttl)
update_remaining(math.max(0, (key_remaining or 0) - estimate))
refresh_rule_telemetry_ttl()
return {1, id, estimate, max_usage, math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[10])}
