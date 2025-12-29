from .SE3_group   import __main__ as SE3_bench
from .SO3_group   import __main__ as SO3_bench
from .SO3_algebra import __main__ as so3_bench
from .SE3_algebra import __main__ as se3_bench

def main_bench():
    SE3_bench.main()
    SO3_bench.main()
    so3_bench.main()
    se3_bench.main()

if __name__ == "__main__":
    main_bench()
