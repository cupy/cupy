try:
    from cudnn import *
except:
    available = False
    enabled = False
