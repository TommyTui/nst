"""
Command line interface for the application.
"""

import os
from nst import style_transfer


def parse_list(s, dtype=int):
    if len(s) >= 1:
        if s[0] == '[' and s[-1] == ']':
            s = s[1:-1]
        elif s[0] == '(' and s[-1] == ')':
            s = s[1:-1]
    if s == '':
        return []
    return [dtype(x) for x in s.split(',')]
    

def make():
    try:
        print('> Content image: ', end='')
        content_image = input()
        print('> Style image: ', end='')
        style_image = input()
        print('> Output location: ', end='')
        output = input()
        print('> Content layer (int): ', end='')
        content_layer = input()
        print('> Content weight (float): ', end='')
        content_weight = input()
        print('> Style layers (int list): ', end='')
        style_layers = input()
        print('> Style weights (float list): ', end='')
        style_weights = input()
        print('> Learning rate (float): ', end='')
        learning_rate = input()
        print('> Iterations (int): ', end='')
        iterations = input()
        print('> Verbose (bool): ', end='')
        verbose = input()
        print('Transferring...')
        style_layers_list = parse_list(style_layers)
        style_weights_list = parse_list(style_weights, float)
        assert len(style_layers_list) == len(style_weights_list)
        params = {
            'content_image': content_image,
            'style_image': style_image,
            'content_layer': int(content_layer) if content_layer != '' else 21,
            'content_weight': float(content_weight) if content_weight != '' else 1e-6,
            'style_layers': style_layers_list if len(style_layers_list) != 0 else (2, 7, 12, 21, 30),
            'style_weights': style_weights_list if len(style_weights_list) != 0 else (9, 9, 9, 9, 9),
            'learning_rate': float(learning_rate) if learning_rate != '' else 8e-2,
            'iterations': int(iterations) if iterations != '' else 180,
            'verbose': bool(verbose) if verbose != '' else True,
            'output': output if output != '' else 'output.jpg' 
            }
        style_transfer(**params)
        print('Done! Output image saved to ' + output)
    except Exception as e:
        print('Error: ' + str(e))

def main():
    print('Enter "make" to create a new style transfer image.')
    print('Enter "q" to quit.')
    print('> ', end='')
    command = input()
    command = command.lower()
    command = command.strip()
    if command == 'make':
        make()
        print('Hit ENTER to continue...')
        input()
        main()
    elif command == 'q':
        pass
    else:
        print('Invalid command.')
        main()


if __name__ == '__main__':
    print('Neural style transfer developed by TommyTui.')
    main()


