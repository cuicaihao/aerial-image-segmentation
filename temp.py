# from contextlib import redirect_stderr
# import io
# import time
# import sys

# from tqdm import tqdm
# from gooey import Gooey, GooeyParser


# @Gooey(progress_regex=r"(\d+)%")
# def main():
#     parser = GooeyParser(prog="example_progress_bar_1")
#     _ = parser.parse_args(sys.argv[1:])

#     f = io.StringIO()
#     with redirect_stderr(f):
#         for i in tqdm(range(21)):
#             prog = f.getvalue().split('\r ')[-1].strip()
#             print(prog)
#             time.sleep(0.2)


# if __name__ == "__main__":
#     sys.exit(main())


from contextlib import redirect_stderr
import io
import time
import sys

from tqdm import tqdm
from gooey import Gooey, GooeyParser


@Gooey(progress_regex=r"(\d+)%")
def main():
    parser = GooeyParser(prog="example_progress_bar_1")
    _ = parser.parse_args(sys.argv[1:])

    progress_bar_output = io.StringIO()
    with redirect_stderr(progress_bar_output):
        for x in tqdm(range(0, 100, 10), file=sys.stdout):
            print(progress_bar_output.read())
            time.sleep(0.2)


if __name__ == "__main__":
    main()
