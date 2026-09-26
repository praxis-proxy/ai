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

-- Reconcile stays O(1): the key's published balance moves by the settled
-- delta and is recomputed exactly on that key's next reservation.
local max_window = 0
local min_capacity = nil
for i = 1, budget_count do
  local window = tonumber(ARGV[4 + (i * 2) - 1])
  local capacity = tonumber(ARGV[4 + (i * 2)])
  if window > max_window then max_window = window end
  if min_capacity == nil or capacity < min_capacity then min_capacity = capacity end
end
if redis.call('ZSCORE', KEYS[10], KEYS[1]) ~= false then
  local previous = tonumber(redis.call('HGET', KEYS[11], KEYS[1]) or '0')
  local next_remaining = math.min(min_capacity or 0, math.max(0, previous + estimate - actual))
  redis.call('INCRBY', KEYS[12], next_remaining - previous)
  redis.call('HSET', KEYS[11], KEYS[1], next_remaining)
end
local telemetry_ttl = math.max(max_window + timeout_ms, 1000)
for i = 8, 12 do redis.call('PEXPIRE', KEYS[i], telemetry_ttl) end
return {1, actual, math.max(0, estimate - actual), math.max(0, actual - estimate), math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[10])}
