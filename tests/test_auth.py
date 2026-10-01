"""Auth core — written test-first: email normalization, the sign-up domain rule, password hashing."""

import pytest

from weather_risk.api.auth import hash_password, normalize_email, signup_email_error, verify_password


# -- email ---------------------------------------------------------------------------

def test_email_is_normalized_before_validation():
    assert normalize_email("  Dana.Levi@MOVEO.CO.IL\n") == "dana.levi@moveo.co.il"
    assert normalize_email("dana＠moveo.co.il") == "dana@moveo.co.il"  # full-width @ folds to "@" (NFKC)


@pytest.mark.parametrize("email", ["dana@moveo.co.il", "dana.levi+alerts@moveo.co.il", " DANA@Moveo.Co.Il "])
def test_moveo_addresses_may_sign_up(email):
    assert signup_email_error(normalize_email(email)) is None


@pytest.mark.parametrize("email", [
    "dana@moveo.co.il.attacker.com",  # domain only starts with moveo.co.il
    "dana@sub.moveo.co.il",           # must end exactly with "@moveo.co.il"
    "dana@xmoveo.co.il",
    "dana@moveo.co.il@evil.com",      # the real domain is the part after the last "@"
    "dana@moveo.co.il.",
    "dana@moveo.co.іl",               # Cyrillic "і" look-alike: NFKC keeps it, so the domain differs
    "dana@gmail.com",
    "@moveo.co.il",
    "da na@moveo.co.il",
    ".dana@moveo.co.il",
    "dana..levi@moveo.co.il",
    '"dana"@moveo.co.il',
    "dana",
    "",
])
def test_other_addresses_may_not_sign_up(email):
    assert signup_email_error(normalize_email(email)) is not None


def test_domain_error_names_the_allowed_domain():
    assert "@moveo.co.il" in signup_email_error("dana@moveo.co.il.attacker.com")


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
