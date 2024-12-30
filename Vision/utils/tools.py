import numpy as np
import random, torch, os, time, json

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

@torch.no_grad()
def save_weights(model_path, model, trainable_names):
    model.eval()
    trainable = {}
    for n, p in model.named_parameters():
        if n in trainable_names:
            trainable[n] = p.data
    torch.save(trainable, model_path)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


class MetricsLogger(object):
    def __init__(self, fname, reinitialize=False, rename_interval=20, suffix='.jsonl'):
        self.base_name = fname
        self.fname = fname+suffix
        self.suffix = suffix
        self.reinitialize = reinitialize
        if self.reinitialize:
            for fn in os.listdir(os.path.dirname(self.base_name)):
                if fn.startswith(self.base_name) and fn.endswith(self.suffix):
                    print('{} exists, deleting...'.format(self.base_name+self.suffix))
                    os.remove(fn)
        self.cur_iter = 1
        self.rename_interval = rename_interval

    def log(self, record=None, **kwargs):
        """
        Assumption: no newlines in the input.
        """
        self.cur_iter += 1
        if record is None:
            record = {}
        record.update(kwargs)
        record['_stamp'] = time.time()
        with open(self.fname, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')


class TrainLogger(object):
    def __init__(self, fname, reinitialize=False, logstyle='%3.3f', buffer_size=1):
        self.root = fname
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.reinitialize = reinitialize
        self.metrics = []
        self.logstyle = logstyle  # One of '%3.3f' or like '%3.3e'
        self.buffer = {}
        self.buffer_size = buffer_size
        self._counter = 0

    # Delete log if re-starting and log already exists
    def reinit(self, item):
        if os.path.exists('%s/%s.log' % (self.root, item)):
            if self.reinitialize:
                # Only print the removal mess
                if 'sv' in item:
                    if not any('sv' in item for item in self.metrics):
                        print('Deleting singular value logs...')
                else:
                    print('{} exists, deleting...'.format('%s_%s.log' % (self.root, item)))
                os.remove('%s/%s.log' % (self.root, item))

    # Log in plaintext; this is designed for being read in MATLAB(sorry not sorry)
    def log(self, itr, **kwargs):
        self._counter += 1
        for arg in kwargs:
            if arg not in self.metrics:
                if self.reinitialize:
                    self.reinit(arg)
                self.metrics += [arg]
            if self.logstyle == 'pickle':
                print('Pickle not currently supported...')
                # with open('%s/%s.log' % (self.root, arg), 'a') as f:
                # pickle.dump(kwargs[arg], f)
            elif self.logstyle == 'mat':
                print('.mat logstyle not currently supported...')
            else:
                key = '%s/%s.log' % (self.root, arg)
                value_str = self.logstyle % kwargs[arg] if type(kwargs[arg]) == float else kwargs[arg]
                value = '%d: %s\n' % (itr, value_str)
                if key in self.buffer:
                    self.buffer[key] += value
                else:
                    self.buffer[key] = value
                # with open('%s/%s.log' % (self.root, arg), 'a') as f:
                #     f.write())
        if self._counter % self.buffer_size == 0:
            self._write()

    def _write(self):
        if self._counter > 0:
            for key in self.buffer:
                with open(key, 'a') as f:
                    f.write(self.buffer[key])
                self.buffer[key] = ''
        self._counter = 0

    def close(self):
        self._write()
