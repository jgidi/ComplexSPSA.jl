"""
    callback_accumulator(xacc)

For internal use only. Returns a callback function to pass to `Qiskit` in order to
accumulate the values of the variables optimized at each iteration.
"""
function callback_accumulator(xacc)
    it = 0
    return function cb!(N, x, fx, dx, is_accepted)
        it += 1
        xacc[:, it] = x
    end
end

"""
    pip_install_dependencies()

Ensure the [Qiskit](https://qiskit.org) library is installed on the
python distribution reached by julia.

Notes
====
* Assumes the `pip` tool is present on the system.
"""
function pip_install_dependencies()
    py"""
    import pip
    pip.main(["install", "qiskit"])
    """
end
