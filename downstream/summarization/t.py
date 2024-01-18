from dataclasses import dataclass


@dataclass
class DT:
    loss: str
    loss2: str

if __name__ == '__main__':
    dt = DT(loss='asa', loss2='asdsd')
    r = isinstance(dt, dict)
    print(r)