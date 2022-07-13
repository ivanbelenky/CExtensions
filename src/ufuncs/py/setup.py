from distutils.core import setup, Extension

def main():
    setup(name="npufunc",
          version="0.0.1",
          description="ufunc examples",
          author="Ivan Belenky",
          author_email="ivanbelenky@gmail.com",
          ext_modules=[
            Extension("npufunc", ["ufuncs.c"])])
if __name__ == "__main__":
    main()
