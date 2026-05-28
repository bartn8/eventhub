import copy
import torch
import torch.distributed as dist


class SummationMeter:
    def __init__(self, string_format=None):
        self.sum = 0
        self.string_format = string_format

    def reset(self):
        self.__init__(string_format=self.string_format)

    def update(self, val, n=1):
        self.sum += val * n

    @property
    def value(self):
        return copy.copy(self.sum)

    def __str__(self):
        return self.string_format % self.value

    def __add__(self, other):
        assert isinstance(other, SummationMeter)

        output = copy.deepcopy(self)
        output.update(val=other.sum)

        return output


class AverageMeter(SummationMeter):
    def __init__(self, string_format=None):
        super().__init__(string_format=string_format)
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        super().update(val=val, n=n)
        self.count += n
        self.avg = self.sum / self.count

    @property
    def value(self):
        return copy.copy(self.avg)

    def __add__(self, other):
        assert isinstance(other, AverageMeter)

        output = copy.deepcopy(self)

        output.update(val=other.avg, n=other.count)

        return output
    
class ListAverageMeter:
    def __init__(self, string_format=None):
        self.values = []
        self.string_format = string_format

    def reset(self):
        self.__init__(string_format=self.string_format)

    def update(self, val, n=1):
        self.values.extend([val] * n)

    @property
    def value(self):
        return sum(self.values) / len(self.values) if len(self.values) > 0 else 0.0
    
    @property
    def count(self):
        return len(self.values)
    
    @property
    def items(self):
        return copy.copy(self.values)

    def __str__(self):
        return self.string_format % self.value

    def __add__(self, other):
        assert isinstance(other, ListAverageMeter)

        output = copy.deepcopy(self)
        output.values.extend(other.values)

        return output


class Metric:
    def __init__(self, average_by='image', string_format=None):
        assert average_by in ['pixel', 'image']
        self.average_meter = ListAverageMeter(string_format=string_format)
        self.average_by = average_by

    def reset(self):
        self.__init__(average_by=self.average_by, string_format=self.average_meter.string_format)

    @torch.no_grad()
    def update(self, pred, ground_truth, mask):
        data_count = 0
        error = 0.0

        for p, gt, m in zip(pred, ground_truth, mask):
            if not m.any():
                continue
            error += self.calculate_error(p, gt, m).to(torch.float).item()
            if self.average_by == 'pixel':
                data_count += m.sum().item()
            elif self.average_by == 'image':
                data_count += 1
            else:
                raise NotImplementedError

        error /= data_count

        self.average_meter.update(val=error, n=data_count)

    def calculate_error(self, pred, ground_truth, mask):
        raise NotImplementedError

    @property
    def value(self):
        return self.average_meter.value
    
    @property
    def items(self):
        return self.average_meter.items

    def __str__(self):
        return str(self.average_meter)

    def __add__(self, other):
        assert isinstance(other, Metric)

        output = copy.deepcopy(self)

        output.average_meter = output.average_meter + other.average_meter

        return output


class EndPointError(Metric):
    def calculate_error(self, pred, ground_truth, mask):
        # pred, ground_truth, mask: (H, W)
        pred, ground_truth = pred[mask], ground_truth[mask]
        error = torch.abs(pred - ground_truth)

        if self.average_by == 'pixel':
            final_error = torch.nansum(error)
        elif self.average_by == 'image':
            final_error = torch.nanmean(error)
        else:
            raise NotImplementedError

        return final_error


class NPixelError(Metric):
    def __init__(self, n=1, average_by='image', string_format=None):
        super().__init__(average_by=average_by, string_format=string_format)
        self.n = n

    def calculate_error(self, pred, ground_truth, mask):
        # pred, ground_truth, mask: (H, W)
        pred, ground_truth = pred[mask], ground_truth[mask]
        error = torch.abs(pred - ground_truth)
        error_mask = error > self.n
        error_mask = error_mask.to(torch.float)

        if self.average_by == 'pixel':
            final_error = torch.nansum(error_mask)
        elif self.average_by == 'image':
            final_error = torch.nanmean(error_mask)
        else:
            raise NotImplementedError

        return final_error * 100.0


class RootMeanSquareError(Metric):
    def calculate_error(self, pred, ground_truth, mask):
        # pred, ground_truth, mask: (H, W)
        pred, ground_truth = pred[mask], ground_truth[mask]
        error = ((pred - ground_truth) ** 2)
        error = torch.nanmean(error).sqrt()

        if self.average_by == 'image':
            final_error = error
        else:
            raise NotImplementedError

        return final_error


class RelError(Metric):
    def calculate_error(self, pred, ground_truth, mask):
        # pred, ground_truth, mask: (H, W)
        pred, ground_truth = pred[mask], ground_truth[mask]
        error = torch.abs(pred - ground_truth) / ground_truth
        

        if self.average_by == 'pixel':
            final_error = torch.nansum(error)
        elif self.average_by == 'image':
            final_error = torch.nanmean(error)
        else:
            raise NotImplementedError

        return final_error

class DeltaError(Metric):
    def __init__(self, n=1, average_by='image', string_format=None):
        super().__init__(average_by=average_by, string_format=string_format)
        self.n = n

    def calculate_error(self, pred, ground_truth, mask):
        # pred, ground_truth, mask: (H, W)

        pred, ground_truth = pred[mask], ground_truth[mask]

        thresh = torch.max(ground_truth / pred, pred / ground_truth)
        error = (thresh < 1.05 ** (self.n)).float()
        

        if self.average_by == 'pixel':
            final_error = error.sum()
        elif self.average_by == 'image':
            final_error = error.mean()
        else:
            raise NotImplementedError

        return final_error
