from distutils.core import setup, Extension

def main():
    setup(name="llist",
          version="0.0.1",
          description="Linked List",
          author="Ivan Belenky",
          author_email="ivanbelenky@gmail.com",
          ext_modules=[Extension("llist", ["list.c"])])

if __name__ == "__main__":
    main()
