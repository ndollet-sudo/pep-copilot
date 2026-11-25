from pepcopilot.eval.agent_eval import check_answer, tools_ok


def test_check_answer_whole_numbers():
    assert check_answer("It is 8 years.", ["8"])
    assert not check_answer("Created in 2018.", ["8"])
    assert check_answer("PEP 622 was Superseded by PEP 634", ["supersed", "634"])
    assert check_answer("I could not find it", ["not|none"])


def test_tools_ok_alternatives():
    assert tools_ok(["search_peps", "calculator"], ["search_peps|get_pep", "calculator"])
    assert not tools_ok(["search_peps"], ["find_peps"])
