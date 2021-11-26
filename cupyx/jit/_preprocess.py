def _strip_lambda(source):
    # Find left most point of lambda function
    splits = source.split('lambda')
    candidates = []
    for i, (l, r) in enumerate(zip(splits[:-1], splits[1:])):
        if (len(l) > 0 and not (l[-1].isalnum() or l[-1] == '_')
                and len(r) > 0 and not (r[0].isalnum() or r[0] == '_')):
            candidates.append(i)
    assert len(candidates) > 0
    if len(candidates) >= 2:
        raise RuntimeError('Parse error: multiple lambda function is found.')
    source = 'lambda'.join(splits[i+1:])

    # Trim arguments
    splits = source.split(':')
    args = [s.strip() for s in splits[0].strip().split(',')]
    source = ':'.join(splits[1:]).strip()

    # Find right most point of lambda function
    depth = 0
    str_char = None
    escaped = False
    for i, c in enumerate(source):
        if str_char is not None:
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == str_char:
                str_char = None
        elif c in ("'", '"'):
            str_char = c
        elif c in ('(', '{', '['):
            depth += 1
        elif c in (')', '}', ']'):
            depth -= 1
        if depth < 0 or (depth == 0 and c == ','):
            return args, source[:i]
    return args, source


def _make_function_str(func_name, args, body):
    args = ', '.join(args)
    return f"""
def {func_name}({args}):
    return {body}
"""
