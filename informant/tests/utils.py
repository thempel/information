import numpy as np

class GenerateTestMatrix(type):
    """
    Class taken from pyemma/coordinates/tests/test_readers.py
    """
    def __new__(mcs, name, bases, attr):
        from functools import partial

        # needed for python2
        class partialmethod(partial):
            def __get__(self, instance, owner):
                if instance is None:
                    return self
                return partial(self.func, instance,
                               *(self.args or ()), **(self.keywords or {}))
        new_test_methods = {}

        test_templates = {k: v for k, v in attr.items() if k.startswith('_test') }
        test_parameters = attr['params']
        for test, params in test_templates.items():
            test_param = test_parameters[test]
            for param_set in test_param:
                func = partialmethod(attr[test], **param_set)
                # only 'primitive' types should be used as part of test name.
                vals_str = '_'.join((str(v) if not isinstance(v, np.ndarray) else 'array' for v in param_set.values()))
                assert '[' not in vals_str, 'this char makes pytest think it has to extract parameters out of the testname.'
                out_name = '{}_{}'.format(test[1:], vals_str)
                new_test_methods[out_name] = func

        attr.update(new_test_methods)
        return type.__new__(mcs, name, bases, attr)
