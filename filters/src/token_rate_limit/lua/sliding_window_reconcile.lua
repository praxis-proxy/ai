-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Praxis Contributors

local value = redis.call('HGET', KEYS[3], ARGV[1])
local rule_active_total = tonumber(redis.call('GET', KEYS[8]) or '0')
local function reported_remaining()
  return math.min(9007199254740991, tonumber(redis.call('GET', KEYS[12]) or '0'))
end
if not value then return {0, math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[10])} end
local sep = string.find(value, '|')
local estimate = tonumber(string.sub(value, 1, sep - 1))
local actual = tonumber(ARGV[2])
local budget_count = tonumber(ARGV[3])
local timeout_ms = tonumber(ARGV[4])
redis.call('HDEL', KEYS[3], ARGV[1])
local active_total = math.max(0, tonumber(redis.call('GET', KEYS[5]) or '0') - 1)
redis.call('SET', KEYS[5], active_total)
rule_active_total = math.max(0, rule_active_total - 1)
redis.call('SET', KEYS[8], rule_active_total)
redis.call('ZREM', KEYS[7], KEYS[1] .. '|' .. ARGV[1])
redis.call('ZREM', KEYS[9], KEYS[1] .. '|' .. ARGV[1])
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
redis.call('ZADD', KEYS[2], now_ms, 'settled:' .. ARGV[1] .. ':' .. actual)

local key_remaining = nil
local max_window = 0
for i = 1, budget_count do
  local window = tonumber(ARGV[4 + (i * 2) - 1])
  local capacity = tonumber(ARGV[4 + (i * 2)])
  if window > max_window then max_window = window end
  redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now_ms - window)
  local settled_sum = 0
  local entries = redis.call('ZRANGE', KEYS[2], now_ms - window, '+inf', 'BYSCORE')
  for j = 1, #entries do
    local amount = string.match(entries[j], ':(%d+)$')
    if amount then settled_sum = settled_sum + tonumber(amount) end
  end
  local active_sum = 0
  local active_values = redis.call('HGETALL', KEYS[3])
  for j = 1, #active_values, 2 do
    local value_sep = string.find(active_values[j + 1], '|')
    active_sum = active_sum + tonumber(string.sub(active_values[j + 1], 1, value_sep - 1))
  end
  local available = math.max(0, capacity - settled_sum - active_sum)
  if key_remaining == nil or available < key_remaining then key_remaining = available end
end
if redis.call('ZSCORE', KEYS[10], KEYS[1]) ~= false then
  local previous = tonumber(redis.call('HGET', KEYS[11], KEYS[1]) or '0')
  local next_remaining = key_remaining or 0
  redis.call('INCRBY', KEYS[12], next_remaining - previous)
  redis.call('HSET', KEYS[11], KEYS[1], next_remaining)
end
local telemetry_ttl = math.max(max_window + timeout_ms, 1000)
for i = 8, 12 do redis.call('PEXPIRE', KEYS[i], telemetry_ttl) end
return {1, actual, math.max(0, estimate - actual), math.max(0, actual - estimate), math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[10])}
