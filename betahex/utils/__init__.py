
def parametrized(dec):
    def layer(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return dec(args[0])

        def repl(f):
            return dec(f, *args, **kwargs)

        return repl
    return layer
