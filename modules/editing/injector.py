from typing import Any, Callable

from modules.inversion.diffusion_inversion import DiffusionInversion


class FunctionInject:
    """Overwrites a object method with a new method.
    """

    def __init__(self, obj: Any, func_name: str, func_new: Callable) -> None:
        """Creates a new instance for function overwriting.

        Args:
            obj (Any): Object to overwrite function in.
            func_name (str): Name of the function to overwrite.
            func_new (Callable): Function to replace with.
        """

        self.obj = obj
        self.func_name = func_name
        self.func_old = getattr(obj, func_name)
        self.func_new = func_new

    def begin(self) -> None:
        """Overwrites function.
        """

        # insert function wrapper self.inject
        setattr(self.obj, self.func_name, self.inject)

    def end(self) -> None:
        """Restores original function.
        """

        setattr(self.obj, self.func_name, self.func_old)

    def inject(self, *args: Any, **kwargs: Any) -> Any:
        """Replacement wrapper for new function.

        Returns:
            Any: (Modified) output of original function.
        """

        # to avoid recursion loops, restore to old function before executing the new function
        self.end()  

        out = self.func_new(*args, **kwargs)

        # reapply new function
        self.begin()
        return out
    

class Injector:
    """Overwrites defined functions in the diffusion inversion instance. Useful for editing.
    """

    def __init__(self, inverter: DiffusionInversion) -> None:
        """Creates a new injector instance for overwriting diffusion inversion methods.

        Args:
            inverter (DiffusionInversion): Diffusion inversion instance to inject functions to.
        """

        self.inverter = inverter
        self.injectable_functions = ["unet", "predict_noise", "step_backward"]  # functions to overwrite
        self.injectors = {}  # injected functions

    def __enter__(self) -> "Injector":
        """Context manager"""
        self.begin()
        return self

    def __exit__(self, *exc) -> bool:
        """Context manager"""
        self.end()
        return False

    def begin(self) -> None:
        """Inject all defined functions.
        """

        assert len(self.injectors) == 0, "Already injected."
        for func_name in self.injectable_functions:
            if hasattr(self, func_name):
                # if defined -> inject
                inj = FunctionInject(self.inverter, func_name, getattr(self, func_name))
                inj.begin()

                # store injector
                self.injectors[func_name] = inj

    def end(self) -> None:
        """Restores original functions.
        """

        for inj in reversed(self.injectors.values()):
            inj.end()

        # clear injectors
        self.injectors = {}