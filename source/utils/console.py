import os

from colorama import Fore, Style

class Colored:
    COLOR_TABLE = {
        '[a]': Fore.LIGHTBLACK_EX,
        '[b]': Fore.BLUE,
        '[c]': Fore.CYAN,
        '[g]': Fore.GREEN,
        '[m]': Fore.MAGENTA,
        '[r]': Fore.RED,
        '[w]': Fore.WHITE,
        '[x]': Fore.LIGHTRED_EX,
        '[y]': Fore.YELLOW,
        '[z]': Fore.RESET,
        '[ ]': Style.RESET_ALL
    }

    def colored(self, value: str):
        for k, v in self.COLOR_TABLE.items():
            value = value.replace(k, v)
        return value

class Console(Colored):
    def clear(self):
        os.system('cls' if os.name=='nt' else 'clear')

    def separator(self):
        term_size = os.get_terminal_size()
        print(Fore.LIGHTBLACK_EX + '─' * term_size.columns + Style.RESET_ALL)

    def print(self, value: str, indent: int = 0):
        print(' ' * indent + self.colored(value) + Style.RESET_ALL)

    def string(self, value: str):
        return self.colored(value)
    
    def error(self, value: str):
        self.print(f'[x]{value}[z]')