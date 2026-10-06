"""Three dark switches accept the same spellings as the others (work-orders #69, R8).

ANALEE_PROVISIONING_ENABLED, ANALEE_ENTITLEMENT_ENFORCED and
ANALEE_FILEIT_PULL_ENABLED turned on only for the exact string "True", while
every other switch in this app accepts 1/true/yes/on. The GM setting "true" or
"1" left them silently off and the deploy log said nothing.
"""
import pytest

ON = ["1", "true", "True", "TRUE", "yes", "on", " on "]
OFF = ["", "0", "false", "False", "no", "off", "maybe"]


def _checks():
    import entitlement
    import fileit_pull
    import provisioning
    return [
        ("ANALEE_PROVISIONING_ENABLED", provisioning.enabled),
        ("ANALEE_ENTITLEMENT_ENFORCED", entitlement.enforcement_enabled),
        ("ANALEE_FILEIT_PULL_ENABLED", fileit_pull.enabled),
    ]


@pytest.mark.parametrize("value", ON)
def test_every_on_spelling_turns_the_switch_on(monkeypatch, value):
    for name, check in _checks():
        monkeypatch.setenv(name, value)
        assert check() is True, f"{name}={value!r} should be ON"


@pytest.mark.parametrize("value", OFF)
def test_every_off_spelling_keeps_the_switch_off(monkeypatch, value):
    for name, check in _checks():
        monkeypatch.setenv(name, value)
        assert check() is False, f"{name}={value!r} should be OFF"


def test_unset_is_off(monkeypatch):
    for name, check in _checks():
        monkeypatch.delenv(name, raising=False)
        assert check() is False


def test_the_shared_helper_is_what_they_use():
    from config import env_flag
    import inspect
    import entitlement
    import fileit_pull
    import provisioning
    for fn in (provisioning.enabled, entitlement.enforcement_enabled, fileit_pull.enabled):
        assert "env_flag(" in inspect.getsource(fn)
    assert env_flag("ANALEE_TEST_FLAG_THAT_IS_NOT_SET") is False
    assert env_flag("ANALEE_TEST_FLAG_THAT_IS_NOT_SET", default=True) is True
