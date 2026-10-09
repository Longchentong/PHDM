import torch
import os


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.selected_epochs = [None for _ in range(runs)]

    def selected_index(self, run, mode='max_acc'):
        rows = self.results[run]
        if mode != 'max_acc':
            return min(range(len(rows)), key=lambda index: rows[index][3])
        if self.selected_epochs[run] is not None:
            return self.selected_epochs[run]
        if getattr(self.info, 'val_tie_break', 0):
            return min(range(len(rows)), key=lambda index: (-rows[index][1], rows[index][3], index))
        return max(range(len(rows)), key=lambda index: rows[index][1])

    def add_result(self, run, result):
        assert len(result) == 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    @staticmethod
    def get_results_string(best_result):
        result_string = ''
        r = best_result[:, 0]
        result_string += f'Highest Train: {r.mean():.2f} ± {r.std():.2f}\t'
        r = best_result[:, 1]
        result_string += f'Validation-selected Test: {r.mean():.2f} ± {r.std():.2f}\t'
        r = best_result[:, 2]
        result_string += f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}\t'
        r = best_result[:, 3]
        result_string += f'  Final Train: {r.mean():.2f} ± {r.std():.2f}\t'
        r = best_result[:, 4]
        result_string += f'   Final Test: {r.mean():.2f} ± {r.std():.2f}'

        return result_string

    def print_statistics(self, run=None, mode='max_acc'):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            ind = self.selected_index(run, mode)

            print_str = f'Run {run + 1:02d}:' + \
                        f'Highest Train: {result[:, 0].max():.2f} ' + \
                        f'Highest Valid: {result[:, 1].max():.2f} ' + \
                        f'Validation-selected Test: {result[ind, 2]:.2f} ' + \
                        f'Chosen epoch: {ind + 1}\n' + \
                        f'Final Train: {result[ind, 0]:.2f} ' + \
                        f'Final Test: {result[ind, 2]:.2f}'
            print(print_str)
            self.test = result[ind, 2]

        else:
            best_results = []
            max_val_epoch = 0
            for run_index, r in enumerate(self.results):
                r = 100 * torch.tensor(r)
                ind = self.selected_index(run_index, mode)
                train1 = r[:, 0].max().item()
                test1 = r[ind, 2].item()
                valid = r[:, 1].max().item()
                train2 = r[ind, 0].item()
                test2 = r[ind, 2].item()
                max_val_epoch = ind
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print_str = f'{len(self.results)} runs: '
            r = best_result[:, 0]
            print_str += f'Highest Train: {r.mean():.2f} ± {r.std():.2f} '
            print_str += f'Highest val epoch:{max_val_epoch}\n'
            r = best_result[:, 1]
            print_str += f'Validation-selected Test: {r.mean():.2f} ± {r.std():.2f} '
            r = best_result[:, 4]
            print_str += f'Final Test: {r.mean():.2f} ± {r.std():.2f}'

            self.test = r.mean()
        return print_str

    def output(self, out_path, info):
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')

    def save(self, params, results, filename):
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(f"{results}\n")
            file.write(f"{params}\n")
            file.write('==' * 50)
            file.write('\n')
            file.write('\n')


def save_result(args, results):
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')
    filename = f'results/{args.dataset}/{args.method}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        # Get a dictionary of arguments and their values
        args_dict = vars(args)
        # Write each argument and its value to the file
        for arg, value in args_dict.items():
            write_obj.write(f"{arg}: {value} ")
        # Write the results
        write_obj.write(f"{results.mean():.2f} $\pm$ {results.std():.2f} \n")
