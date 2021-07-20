import csv
import unittest

from metric_handlers import MetricsToCsv
from metric_handlers import Metric

BAD_HEADERS_ERROR = "Wrong headers"
BAD_DATA_ERROR = "Read bad data"


class MetricHandlersTest(unittest.TestCase):
    def testCsvExporter(self):
        metric1 = Metric[str]("metric1")
        metric2 = Metric[float]("metric2")
        metric3 = Metric[int]("metric3")

        metricToCsv = MetricsToCsv("./tests/test.csv", [metric1, metric2, metric3])

        for index in range(2000):
            metric1.Call(index.__str__())
            metric2.Call(float(index))
            metric3.Call(index)

        metricToCsv.Write()

        with open("./tests/test.csv") as file:
            data = csv.reader(file, delimiter=",", quotechar="|")
            index = -2
            for row in data:
                index += 1

                if index == -1:
                    self.assertEqual(row[0], "metric1", msg=BAD_HEADERS_ERROR)
                    self.assertEqual(row[1], "metric2", msg=BAD_HEADERS_ERROR)
                    self.assertEqual(row[2], "metric3", msg=BAD_HEADERS_ERROR)
                    continue

                self.assertEqual(row[0], metric1.data[index], msg=BAD_DATA_ERROR)
                self.assertEqual(float(row[1]), metric2.data[index], msg=BAD_DATA_ERROR)
                self.assertEqual(int(row[2]), metric3.data[index], msg=BAD_DATA_ERROR)
