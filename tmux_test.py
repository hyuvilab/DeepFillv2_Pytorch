'''

Terminal Multiplexer Commands
-----------------------------
    $ tmux new -s <session-name> -n <window-name>
    $ tmux ls
    $ tmux attach -t <session name>
    $ tmux kill-session -t <session name>

    $ Ctrl-b d
        => Exit session
    $ Ctrl-b $
        => Change session name


    [*] ref: https://edykim.com/ko/post/tmux-introductory-series-summary/#tmux-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0

'''
from time import sleep



time_index = 0
while(True):
    print('>> Testing tmux, time index: {}'.format(time_index))
    time_index += 1
    sleep(1)
