from matplotlib import pyplot as pp
import numpy as np

def read_log_file(fname):
    f = open(fname, 'r')
    inxs = []
    errs = []
    for line in f:
        line = line[:-1]
        if 'test' in line:
            err = float(line[-10:-1])
            inx = int(line[10:10+4].replace(',', ''))
            errs.append(err)
            inxs.append(inx)
    f.close()
    return inxs, errs
def plot_err(fnames):
    inxss = []
    errss = []
    pp.xlim(0,100)
    pp.ylim(1,8.0)
    #mng = pp.get_current_fig_manager()
    #mng.frame.Maximize(True)
    pp.xlabel('iterations')
    pp.ylabel('classification error %')

    for fname in fnames:
        inxs, errs = read_log_file('logs/'+fname+'.log')
        inxss.append(inxs)
        errss.append(errs)
    for i in range(len(inxss)-1):
        pp.plot(inxss[i], errss[i], alpha=0.3)
    pp.plot(inxss[-1], errss[-1])
    pp.legend(fnames)
    pp.show()
if __name__ == "__main__":
    fnames = ['ce', 'ae', 'l1','l2', 'dropout1', 'dropout5']
    #, 'ae', 'l1', 'l2', 'dropout1', 'dropout5','dnn2','dnn2_dropout','dnn3','dnn3_dropout']
    plot_err(fnames)
    