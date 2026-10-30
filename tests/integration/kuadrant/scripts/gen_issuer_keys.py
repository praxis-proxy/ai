#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["cryptography>=42"]
# ///
"""Generate an RS256 issuer keypair + a JWKS document for the user-identity IdP.

Writes into the output dir:
  issuer_priv.pem   RSA private key (signs user access tokens)
  issuer_kid.txt    the key id
  jwks.json         {"keys":[{kty,use,alg,kid,n,e}]}  (served at jwksUrl)

Usage: uv run gen_issuer_keys.py --out <dir>
"""
import argparse, base64, json, os, uuid
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def int_to_b64u(n: int) -> str:
    length = (n.bit_length() + 7) // 8
    return b64u(n.to_bytes(length, "big"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    kid = "issuer-" + uuid.uuid4().hex[:8]

    priv_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with open(os.path.join(args.out, "issuer_priv.pem"), "wb") as f:
        f.write(priv_pem)
    with open(os.path.join(args.out, "issuer_kid.txt"), "w") as f:
        f.write(kid)

    pub = key.public_key().public_numbers()
    jwks = {"keys": [{
        "kty": "RSA", "use": "sig", "alg": "RS256", "kid": kid,
        "n": int_to_b64u(pub.n), "e": int_to_b64u(pub.e),
    }]}
    with open(os.path.join(args.out, "jwks.json"), "w") as f:
        json.dump(jwks, f)
    print("wrote issuer_priv.pem, issuer_kid.txt, jwks.json  kid=" + kid)


if __name__ == "__main__":
    main()
