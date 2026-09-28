-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Praxis Contributors

local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local timeout_ms = tonumber(ARGV[3])
local max_keys = tonumber(ARGV[4])
local max_active = tonumber(ARGV[5])
local estimate = tonumber(ARGV[6])

local active_total = tonumber(redis.call('GET', KEYS[4]) or '0')
local rule_active_total = tonumber(redis.call('GET', KEYS[7]) or '0')
local function reported_remaining()
  return math.min(9007199254740991, tonumber(redis.call('GET', KEYS[11]) or '0'))
end

local function update_remaining(value)
  if redis.call('ZSCORE', KEYS[3], KEYS[1]) == false then return end
  local previous = tonumber(redis.call('HGET', KEYS[10], KEYS[1]) or '0')
  redis.call('INCRBY', KEYS[11], value - previous)
  redis.call('HSET', KEYS[10], KEYS[1], value)
end

-- Expired entries are drained a batch at a time so one admission never
-- blocks the server on a large backlog; later admissions finish the job.
local SWEEP_BATCH = 128

local expired_rule = redis.call('ZRANGE', KEYS[8], '-inf', now_ms, 'BYSCORE', 'LIMIT', 0, SWEEP_BATCH)
for i = 1, #expired_rule do
  local member = expired_rule[i]
  local split = string.find(member, '|')
  if split then
    local physical = string.sub(member, 1, split - 1)
    local reservation = string.sub(member, split + 1)
    local active_key = physical .. ':active'
    if redis.call('HDEL', active_key, reservation) > 0 then
      active_total = math.max(0, active_total - 1)
    end
    rule_active_total = math.max(0, rule_active_total - 1)
  end
  redis.call('ZREM', KEYS[6], member)
  redis.call('ZREM', KEYS[8], member)
end
redis.call('SET', KEYS[7], rule_active_total)

local expired_global = redis.call('ZRANGE', KEYS[6], '-inf', now_ms, 'BYSCORE', 'LIMIT', 0, SWEEP_BATCH)
for i = 1, #expired_global do
  local member = expired_global[i]
  local split = string.find(member, '|')
  if split then
    local physical = string.sub(member, 1, split - 1)
    local reservation = string.sub(member, split + 1)
    local active_key = physical .. ':active'
    local value = redis.call('HGET', active_key, reservation)
    if value then
      redis.call('HDEL', active_key, reservation)
      active_total = math.max(0, active_total - 1)
      -- Tokens for an abandoned reservation stay charged (already
      -- decremented at reserve time under the immediate-decrement
      -- design): no credit-back happens on expiry, only on reconcile.
    end
  end
  redis.call('ZREM', KEYS[6], member)
end
redis.call('SET', KEYS[4], active_total)

local expired_keys = redis.call('ZRANGE', KEYS[9], '-inf', now_ms, 'BYSCORE', 'LIMIT', 0, SWEEP_BATCH)
for i = 1, #expired_keys do
  local previous = tonumber(redis.call('HGET', KEYS[10], expired_keys[i]) or '0')
  redis.call('INCRBY', KEYS[11], -previous)
  redis.call('HDEL', KEYS[10], expired_keys[i])
  redis.call('ZREM', KEYS[9], expired_keys[i])
end

local state = redis.call('HMGET', KEYS[1], 'tokens', 'last_refill_ms')
local tokens = tonumber(state[1])
local last_refill_ms = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  last_refill_ms = now_ms
end
local elapsed_ms = math.max(0, now_ms - last_refill_ms)
tokens = math.min(capacity, tokens + (elapsed_ms / 1000.0) * refill_rate)

local ttl = math.max(math.ceil((capacity / refill_rate) * 1000) + timeout_ms, 1000)
local function refresh_rule_telemetry_ttl()
  for i = 7, 11 do redis.call('PEXPIRE', KEYS[i], ttl) end
end
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now_ms)
local key_exists = redis.call('EXISTS', KEYS[1]) == 1
if not key_exists and redis.call('ZCARD', KEYS[3]) >= max_keys then
  refresh_rule_telemetry_ttl()
  return {0, 1, math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[9])}
end
if active_total >= max_active then
  if key_exists then
    redis.call('HSET', KEYS[1], 'tokens', tokens, 'last_refill_ms', now_ms)
    redis.call('PEXPIRE', KEYS[1], ttl)
    update_remaining(math.floor(tokens))
  end
  refresh_rule_telemetry_ttl()
  return {0, 1, math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[9])}
end
if tokens < estimate then
  local deficit = estimate - tokens
  local retry_after_ms = math.max(1, math.ceil((deficit / refill_rate) * 1000))
  redis.call('HSET', KEYS[1], 'tokens', tokens, 'last_refill_ms', now_ms)
  redis.call('PEXPIRE', KEYS[1], ttl)
  update_remaining(math.floor(tokens))
  refresh_rule_telemetry_ttl()
  return {0, retry_after_ms, math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[9])}
end

tokens = tokens - estimate
local usage_after = capacity - tokens
local id = redis.call('INCR', KEYS[5])
redis.call('HSET', KEYS[2], id, estimate .. '|' .. now_ms)
redis.call('INCR', KEYS[4])
rule_active_total = rule_active_total + 1
redis.call('SET', KEYS[7], rule_active_total)
redis.call('ZADD', KEYS[6], now_ms + timeout_ms, KEYS[1] .. '|' .. id)
redis.call('ZADD', KEYS[8], now_ms + timeout_ms, KEYS[1] .. '|' .. id)
redis.call('HSET', KEYS[1], 'tokens', tokens, 'last_refill_ms', now_ms)
redis.call('ZADD', KEYS[3], now_ms + ttl, KEYS[1])
redis.call('ZADD', KEYS[9], now_ms + ttl, KEYS[1])
redis.call('PEXPIRE', KEYS[1], ttl)
redis.call('PEXPIRE', KEYS[2], ttl)
update_remaining(math.floor(tokens))
refresh_rule_telemetry_ttl()
return {1, id, estimate, usage_after, math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[9])}
