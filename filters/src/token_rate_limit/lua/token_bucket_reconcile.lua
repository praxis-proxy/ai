-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Praxis Contributors

local capacity = tonumber(ARGV[3])
local refill_rate = tonumber(ARGV[4])
local timeout_ms = tonumber(ARGV[5])
local telemetry_ttl = math.max(math.ceil((capacity / refill_rate) * 1000) + timeout_ms, 1000)

-- Reconciliation must use the same accounting interpretation as reserve and
-- must reject skew before deleting an active reservation or changing totals.
local config_fingerprint = ARGV[6]
local stored_fingerprint = redis.call('GET', KEYS[12])
if stored_fingerprint and stored_fingerprint ~= config_fingerprint then return {3} end
if not stored_fingerprint then redis.call('SET', KEYS[12], config_fingerprint, 'NX') end

local function extend_expiry(key, ttl)
  redis.call('PEXPIRE', key, ttl, 'NX')
  redis.call('PEXPIRE', key, ttl, 'GT')
end

local value = redis.call('HGET', KEYS[2], ARGV[1])
local rule_active_total = tonumber(redis.call('GET', KEYS[7]) or '0')
local function reported_remaining()
  return math.min(9007199254740991, tonumber(redis.call('GET', KEYS[11]) or '0'))
end
if not value then return {0, math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[9])} end
local sep = string.find(value, '|')
local estimate = tonumber(string.sub(value, 1, sep - 1))
local actual = tonumber(ARGV[2])
redis.call('HDEL', KEYS[2], ARGV[1])
local active_total = math.max(0, tonumber(redis.call('GET', KEYS[4]) or '0') - 1)
redis.call('SET', KEYS[4], active_total, 'KEEPTTL')
rule_active_total = math.max(0, rule_active_total - 1)
redis.call('SET', KEYS[7], rule_active_total, 'KEEPTTL')
redis.call('ZREM', KEYS[6], KEYS[1] .. '|' .. ARGV[1])
redis.call('ZREM', KEYS[8], KEYS[1] .. '|' .. ARGV[1])

local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local state = redis.call('HMGET', KEYS[1], 'tokens', 'last_refill_ms')
local tokens = tonumber(state[1])
local last_refill_ms = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  last_refill_ms = now_ms
end
local elapsed_ms = math.max(0, now_ms - last_refill_ms)
tokens = math.min(capacity, tokens + (elapsed_ms / 1000.0) * refill_rate)

local refund = math.max(0, estimate - actual)
local overage = math.max(0, actual - estimate)
if refund > 0 then
  tokens = math.min(capacity, tokens + refund)
elseif overage > 0 then
  tokens = math.max(0, tokens - overage)
end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'last_refill_ms', now_ms)
if redis.call('ZSCORE', KEYS[9], KEYS[1]) ~= false then
  local previous = tonumber(redis.call('HGET', KEYS[10], KEYS[1]) or '0')
  local next_remaining = math.floor(tokens)
  redis.call('INCRBY', KEYS[11], next_remaining - previous)
  redis.call('HSET', KEYS[10], KEYS[1], next_remaining)
end
extend_expiry(KEYS[1], telemetry_ttl)
extend_expiry(KEYS[2], telemetry_ttl)
for i = 7, 11 do extend_expiry(KEYS[i], telemetry_ttl) end
return {1, actual, refund, overage, math.floor(reported_remaining()), rule_active_total, redis.call('ZCARD', KEYS[9])}
