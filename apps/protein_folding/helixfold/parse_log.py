import os
import argparse


class TrainStatsUnit:
    def __init__(self, filename):
        # 002_AMP_O2-demo_initial_N1C1_dp1_dap1_bp1-bf16-O1-1.txt
        basename = os.path.basename(filename)
        self.filename = basename
        fields = basename.replace(".txt", "").split("-")
        self.opt_version = fields[0]
        self.exp_name = fields[1]
        self.precision = fields[2]
        self.amp_level = fields[3]
        self.run_id = int(fields[4])
        self.batch_costs = []
        self.losses = []
        self.avg_losses = []
        self.tm_score = 0.0

    def update(self, batch_cost, loss, avg_loss):
        self.batch_costs.append(batch_cost)
        self.losses.append(loss)
        self.avg_losses.append(avg_loss)

    def __str__(self):
        assert len(self.batch_costs) == len(self.avg_losses) and len(self.batch_costs) == len(self.losses)
        train_step = len(self.batch_costs)
        res = "{}: train_step={}".format(self.filename, train_step)
        if train_step > 0:
            res += ", batch_cost={:.5f}s, loss={:.6f}, avg_loss={:.6f}".format(self.batch_costs[-1], self.losses[-1], self.avg_losses[-1])
        res += ", tm_score={:.4f}".format(self.tm_score)
        return res


def parse_amp_level(line):
    words = line.strip().split(",")
    amp_level = None
    for i in range(len(words)):
        if "amp_level" in words[i]:
            amp_level = words[i].split("=")[-1]
            amp_level = amp_level.replace("'", "")
    return amp_level


def parse_avg_loss(line):
    # 2023-05-10 14:45:54 INFO [Main] Train_step: 1, loss: 9.173592, reader_cost: 6.24941s, forward_cost: 6.39348s, backward_cost: 1.71375s, gradsync_cost: 0.00000s, update_cost: 0.00387s, batch_cost: 14.36050s, avg_loss: 9.173592, protein_sum: 1, train_cost_sum: 14.36050, ips: 0.06964 protein/s
    words = line.strip().split(" ")
    batch_cost, loss, avg_loss = None, None, None
    for i in range(len(words)):
        if words[i] == "batch_cost:":
            batch_cost = float(words[i + 1].replace("s,", ""))
        elif words[i] == "loss:":
            loss = float(words[i + 1].replace(",", ""))
        elif words[i] == "avg_loss:":
            avg_loss = float(words[i + 1].replace(",", ""))
    return batch_cost, loss, avg_loss


def parse_tm_score(line):
    words = line.strip().split(" ")
    tm_score = None
    for i in range(len(words)):
        if words[i] == "'TM-score':":
            tm_score = float(words[i + 1].replace(",", ""))
            break
    return tm_score


def parse_log(filename):
    filename = os.path.abspath(filename)

    train_stats = TrainStatsUnit(filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "avg_loss:" in line:
                batch_cost, loss, avg_loss = parse_avg_loss(line)
                train_stats.update(batch_cost, loss, avg_loss)
            elif "step:104 valid:" in line:
                train_stats.tm_score = parse_tm_score(line)
            elif "Namespace(amp_level" in line:
                amp_level = parse_amp_level(line)
                assert amp_level == train_stats.amp_level, f"amp_level is not consistent in {filename}."
    return train_stats


def main(logs_dir):
    filenames = []
    if os.path.isdir(args.logs_path):
        for name in os.listdir(args.logs_path):
            if ".swp" not in name:
                filenames.append(args.logs_path + "/" + name)
        print(f"-- There are {len(filenames)} logs under {args.logs_path}")
    elif os.path.isfile(args.logs_path):
        filenames.append(args.logs_path)
        print(f"-- Parsing {args.logs_path}")

    #train_stats_dict = {}
    for filename in sorted(filenames):
        train_stats_unit = parse_log(filename)
        #train_stats_dict[filename] = train_stats_unit
        print(train_stats_unit)
        if args.print_detail:
            print(f"loss={train_stats_unit.losses}")
    print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs_path', type=str, default=None)
    parser.add_argument('--print_detail', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
