import cv2
import random
import torch as tor
import numpy as np




""" Parameters """
CAL_ACC_PERIOD = 500    # steps
SHOW_LOSS_PERIOD = 100   # steps
SAVE_MODEL_PERIOD = 1000   # epochs
SAVE_JSON_PERIOD = 50  # steps

AVAILABLE_SIZE = None
EVAL_TRAIN_SIZE = 100
EVAL_TEST_SIZE = 25

EPOCH = 30
STEP = 50000
BATCHSIZE = 1
LR = 0.0001
LR_STEPSIZE, LR_GAMMA = 5000, 0.95




class Trainer :
    def __init__(self, recorder, base_train, novel_support, novel_test, shot, way, cpu=False, lr=LR, step=None) :
        self.recorder = recorder
        self.base_train = base_train
        self.novel_support = novel_support
        self.novel_test = novel_test
        self.way = way
        self.shot = shot
        self.cpu = cpu
        self.lr = lr
        self.step = step if step else STEP

        self.model = self.recorder.models["relationnet"]
        self.model.way, self.model.shot = self.way, self.shot
        if not self.cpu :
            self.model.cuda()

        # optim = tor.optim.SGD(model.fc_1.parameters(), lr=LR)
        self.optim = tor.optim.Adam(self.model.parameters(), lr=self.lr)
        #self.loss_func = tor.nn.CrossEntropyLoss().cuda()
        self.loss_fn = tor.nn.MSELoss().cuda()
        self.lr_schedule = tor.optim.lr_scheduler.StepLR(optimizer=self.optim, step_size=LR_STEPSIZE, gamma=LR_GAMMA)



    def dump_novel_train(self) :
        way_pick = random.sample(range(self.base_train.shape[0]), self.way - 1)
        shot_pick = random.sample(range(self.base_train.shape[1]), self.shot + 5)

        x = self.base_train[way_pick][:, shot_pick[:-5]]

        novel_pick = random.randrange(20)
        x = np.vstack((x, self.novel_support[novel_pick].reshape(1, 5, 32, 32, 3)))
        x_query = self.base_train[way_pick][:, shot_pick[-5:]]
        x_query = np.vstack((x_query, self.novel_support[novel_pick].reshape(1, 5, 32, 32, 3)))
        y_query = np.array([i // 25 for i in range(self.way * 25)])

        return  x, x_query, y_query



    def eval(self) :
        self.model.way = 20
        self.model.shot = 5
        self.novel_support_tr = tor.Tensor(self.novel_support).permute(0, 1, 4, 2, 3).cuda()
        correct, total = 0, self.novel_test.shape[0] * EVAL_TEST_SIZE
        self.model.training = False

        for label_idx, data in enumerate(self.novel_test) :
            for img in data[:EVAL_TEST_SIZE] :
                img = tor.Tensor(img).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                scores = self.model(self.novel_support_tr, img)
                pred = int(tor.argmax(scores))
                if pred == label_idx :
                    correct += 1

        self.model.training = True
        self.model.way = self.way
        self.model.shot = self.shot

        return correct / total



    def train(self) :
        loss_list = []
        train_acc_list = []

        for i in range(self.step) :
            print("|Steps: {:>5} |".format(self.recorder.get_steps()), end="\r")
            self.optim.zero_grad()

            x, x_query, y_query_idx = self.dump_novel_train()
            x = tor.Tensor(x).permute(0, 1, 4, 2, 3)
            x_query = tor.Tensor(x_query).unsqueeze(0).permute(0, 3, 1, 2) if x_query.ndim == 3 else tor.Tensor(x_query).view(25, 32, 32, 3).permute(0, 3, 1, 2)
            y_query = tor.Tensor((np.array(y_query_idx) == np.array((list(range(5))*25))).astype(np.uint8))
            y_query = y_query.view(y_query.size(0), 1)

            if not self.cpu :
                x, x_query, y_query = x.cuda(), x_query.cuda(), y_query.cuda()

            scores = self.model(x, x_query)

            # calculate training accuracy
            acc = tor.argmax(scores.view(25, 5), dim=1) == tor.LongTensor(np.array([i // 5 for i in range(25)])).cuda()
            acc = np.mean(acc.cpu().numpy())
            train_acc_list.append(acc)

            loss = self.loss_fn(scores, y_query)
            loss.backward()

            loss_list.append(float(loss.data))


            if self.recorder.get_steps() % SHOW_LOSS_PERIOD == 0 :
                loss_avg = round(float(np.mean(np.array(loss_list))), 6)
                train_acc_avg = round(float(np.mean(np.array(train_acc_list))), 5)
                self.recorder.checkpoint({
                    "loss": loss_avg,
                    "train_acc": train_acc_avg
                })
                print("|Loss: {:<8} |Train Acc: {:<8}".format(loss_avg, train_acc_avg))

                loss_list = []
                train_acc_list = []

            if self.recorder.get_steps() % SAVE_JSON_PERIOD == 0 :
                self.recorder.save_checkpoints()

            if self.recorder.get_steps() % SAVE_MODEL_PERIOD == 0 :
                self.recorder.save_models()

            if self.recorder.get_steps() % CAL_ACC_PERIOD == 0 :
                acc = self.eval()
                self.recorder.checkpoint({
                    "acc": acc,
                    "lr": self.optim.param_groups[0]["lr"]
                })
                print("|Acc: {:<8}".format(round(acc, 5)))


            self.optim.step()
            self.lr_schedule.step()
            self.recorder.step()


