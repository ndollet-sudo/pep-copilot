from pepcopilot.agent.tools import calculator, validate_args


SCHEMA = {"type": "object", "properties": {"query": {"type": "string"},
                                           "k": {"type": "integer", "minimum": 1, "maximum": 10}},
          "required": ["query"]}


def test_validate_args_cases():
    assert validate_args(SCHEMA, '{"query": "gil"}') == (True, {"query": "gil"}, "")
    ok, _, err = validate_args(SCHEMA, "{}")
    assert not ok and "query" in err
    ok, _, err = validate_args(SCHEMA, "not json")
    assert not ok and "JSON" in err
    ok, _, err = validate_args(SCHEMA, '{"query": 3}')
    assert not ok and "string" in err
    ok, _, err = validate_args(SCHEMA, '{"query": "x", "k": 99}')
    assert not ok and "<= 10" in err
    ok, args, _ = validate_args(SCHEMA, '{"query": "x", "junk": 1, "k": "3"}')
    assert ok and args == {"query": "x", "k": 3}           # unknown filtered, "3" coerced


def test_calculator_is_safe_and_exact():
    assert calculator("4739 * 812").observation == "3848068"
    assert calculator("(2023 - 2010)").observation == "13"
    for bad in ("__import__('os')", "2 ^ 3", "9 ** 999999"):
        try:
            calculator(bad)
            raise AssertionError(f"{bad} should be rejected")
        except (ValueError, SyntaxError):
            pass


def test_dispatch_never_raises(registry):
    r = registry.dispatch("serach_peps", '{"query": "x"}')
    assert not r.ok and r.error_type == "unknown_tool" and "search_peps" in r.observation
    r = registry.dispatch("calculator", "{}")
    assert r.error_type == "invalid_args" and "expression" in r.observation
    r = registry.dispatch("calculator", '{"expression": "1/0"}')
    assert r.error_type == "tool_error"
    r = registry.dispatch("search_peps", '{"query": "nothing at all"}')
    assert r.ok and r.error_type == "empty"


def test_metadata_tools_on_real_corpus(registry):
    r = registry.dispatch("get_pep", '{"number": 572}')
    assert '"python_version": "3.8"' in r.observation and r.peps == (572,)
    r = registry.dispatch("get_pep", '{"number": 9999}')
    assert "not in the knowledge base" in r.observation
    r = registry.dispatch("find_peps", '{"type": "Process"}')
    assert r.observation.startswith("count: 36")
    r = registry.dispatch("find_peps", '{"status": "Final", "type": "Standards Track", "python_version": "3.11"}')
    assert r.observation.startswith("count: 13")
