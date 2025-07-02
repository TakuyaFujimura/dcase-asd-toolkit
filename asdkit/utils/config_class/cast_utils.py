def cast_str(v) -> str:
    if isinstance(v, str):
        return v
    elif isinstance(v, (int, float)):
        return str(v)
    else:
        raise ValueError("Cannot cast to string")


def check_dcase(v):
    if v in [f"dcase202{i}" for i in range(6)]:
        return v
    else:
        raise ValueError("Unexpected dcase type")
