from pynput import mouse


def on_move(x, y):
    print('Pointer moved to {o}'.format(
        (x, y)))


def on_click(x, y, button, pressed):
    print('{0} at {1}'.format('Pressed' if pressed else 'Released', (x, y)))
    print(button)
    if not pressed:
        return False


def on_scroll(x, y, dx, dy):
    print('scrolled {0} at {1}'.format(
        'down' if dy < 0 else 'up',
        (x, y)))


while True:
    with mouse.Listener(no_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
        listener.join()