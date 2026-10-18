"""Auth core — written test-first: email normalization, the sign-up domain rule, password hashing."""

import pytest

from weather_risk.api.auth import hash_password, normalize_email, signup_email_error, verify_password


# -- email ---------------------------------------------------------------------------

def test_email_is_normalized_before_validation():
    assert normalize_email("  Dana.Levi@EXAMPLE.CO.IL\n") == "dana.levi@example.co.il"
    assert normalize_email("dana＠example.co.il") == "dana@example.co.il"  # full-width @ folds to "@" (NFKC)


@pytest.mark.parametrize("email", ["dana@example.co.il", "dana.levi+alerts@example.co.il", " DANA@Example.Co.Il "])
def test_domain_addresses_may_sign_up(email):
    assert signup_email_error(normalize_email(email)) is None


@pytest.mark.parametrize("email", [
    "dana@example.co.il.attacker.com",  # domain only starts with example.co.il
    "dana@sub.example.co.il",           # must end exactly with "@example.co.il"
    "dana@xexample.co.il",
    "dana@example.co.il@evil.com",      # the real domain is the part after the last "@"
    "dana@example.co.il.",
    "dana@example.co.іl",               # Cyrillic "і" look-alike: NFKC keeps it, so the domain differs
    "dana@gmail.com",
    "@example.co.il",
    "da na@example.co.il",
    ".dana@example.co.il",
    "dana..levi@example.co.il",
    '"dana"@example.co.il',
    "dana",
    "",
])
def test_other_addresses_may_not_sign_up(email):
    assert signup_email_error(normalize_email(email)) is not None


def test_domain_error_names_the_allowed_domain():
    assert "@example.co.il" in signup_email_error("dana@example.co.il.attacker.com")


# -- passwords -----------------------------------------------------------------------

def test_password_hash_is_salted_and_never_contains_the_password():
    first, second = hash_password("12345678"), hash_password("12345678")
    assert first != second  # random salt
    assert "12345678" not in first and first.startswith("scrypt$")


def test_verify_password():
    stored = hash_password("correct horse battery")
    assert verify_password("correct horse battery", stored)
    assert not verify_password("correct horse batterY", stored)
    assert not verify_password("", stored)


@pytest.mark.parametrize("stored", ["", "plaintext", "scrypt$broken", "bcrypt$2b$12$abc", "scrypt$x$8$1$AAAA$AAAA"])
def test_malformed_hash_never_verifies(stored):
    assert not verify_password("12345678", stored)
