import argparse
tmpstr =  str(1.2)+" "+str(2.2)
def main():
    parser = argparse.ArgumentParser(description='Process a list of floats.')
    parser.add_argument('floats', metavar='F', type=float,default=tmpstr, nargs='+', help='a list of float numbers')

    args = parser.parse_args()
    float_list = args.floats

    print('List of floats:', float_list)
    for elem in float_list:
        elem = elem+1
        print(elem)

if __name__ == '__main__':
    main()
