print("""Please select a connector:
1. llama.cpp
2. ctransformers""")
choice = input('Enter your choice (1 to 2): ')
if choice == '1':
    print(
        """Connector: llama.cpp is selected!
Please select an interface:
1. graphical
2. command-line"""
        )
    option = input('Enter your choice (1 to 2): ')
    if option == '1':
        from .cpp import *
    elif option == '2':
        from .gpp import *
    else:
        print('Not a valid number.')
elif choice == '2':
    print(
        """Connector: ctransformers is selected!
Please select an interface:
1. graphical
2. command-line"""
        )
    option = input('Enter your choice (1 to 2): ')
    if option == '1':
        from .c import *
    elif option == '2':
        from .g import *
    else:
        print('Not a valid number.')
else:
    print('Not a valid number.')