from distutils.core import setup, Extension

def main():
    setup(name="lreg",
          version="0.0.1",
          description="Linear Regression methods",
          author="Ivan Belenky",
          author_email="ivanbelenky@gmail.com",
          ext_modules=[
            Extension("lreg", ["lreg.c"])])
if __name__ == "__main__":
    main()
