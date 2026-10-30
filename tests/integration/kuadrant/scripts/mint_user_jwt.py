#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["pyjwt>=2.8", "cryptography>=42"]
# ///
"""Mint an RS256 user access JWT for the spike IdP.

Models a Keycloak-shaped token: `resource_access` maps a server name to a roles
list, e.g. {"weather":{"roles":["tool:get_forecast"]}}. Authorino validates
signature (via JWKS), issuer, audience, expiry.

Usage (via uv):
  uv run mint_user_jwt.py --key issuer_priv.pem --kid <kid> --sub userA \
     --iss http://jwks:8089 --aud mcp-gateway --exp 300 \
     --resource-access '{"weather":{"roles":["tool:get_forecast"]}}'
Flags: --tamper (break signature), --exp may be negative (already expired).
"""
import argparse, json, sys, time
import jwt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--kid", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--iss", required=True)
    ap.add_argument("--aud", required=True)
    ap.add_argument("--exp", type=int, default=300)
    ap.add_argument("--resource-access", default="{}")
    ap.add_argument("--tamper", action="store_true")
    args = ap.parse_args()

    with open(args.key) as f:
        priv = f.read()

    now = int(time.time())
    payload = {
        "iss": args.iss,
        "aud": args.aud,
        "sub": args.sub,
        "preferred_username": args.sub,
        "iat": now,
        "exp": now + args.exp,
        "resource_access": json.loads(args.resource_access),
    }
    token = jwt.encode(payload, priv, algorithm="RS256", headers={"kid": args.kid})

    if args.tamper:
        head, mid, sig = token.split(".")
        # Flip the FIRST signature char (6 fully-significant bits) to a guaranteed
        # different value — reliably invalidates the RS256 signature. (The last char
        # encodes only padding-adjacent bits, so changing it is not reliable.)
        bad = ("B" if sig[0] != "B" else "C") + sig[1:]
        token = f"{head}.{mid}.{bad}"

    sys.stdout.write(token)


if __name__ == "__main__":
    main()
