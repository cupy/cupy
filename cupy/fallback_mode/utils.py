"""
Utilities needed for fallback_mode.

TODO: call_cupy()
TODO: call_numpy()
"""


class FallbackUtil:

    attr_list = []

    def __init__(self):
        self.notifications = True

    def notification_status(self):
        return self.notifications

    def set_notification_status(self, status):
        self.notifications = status
        print("Notification status is now {}".format(self.notifications))

    @classmethod
    def clear_attrs(cls):
        cls.attr_list = []

    @classmethod
    def add_attrs(cls, attr):
        cls.attr_list.append(attr)

    @classmethod
    def get_attr_list_copy(cls):
        return cls.attr_list.copy()


def get_last_and_rest(attr_list):
    path = ".".join(attr_list[:-1])
    return path, attr_list[-1]


def join_attrs(attr_list):
    path = ".".join(attr_list)
    return path
