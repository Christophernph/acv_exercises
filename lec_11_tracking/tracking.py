import os

if __name__ == '__main__':
    
    abs_dir = os.path.dirname( os.path.abspath(__file__))
    dataset = os.path.join(abs_dir, '../data/tracking/')