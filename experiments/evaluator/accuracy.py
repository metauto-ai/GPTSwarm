
class Accuracy:
    def __init__(self):
        self._num_correct = 0
        self._num_total = 0
    
    def update(self, predicted: str, target: str) -> None:
        is_correct = predicted == target
        self._num_correct += int(is_correct)
        self._num_total += 1

    def get(self) -> float:
        return self._num_correct / self._num_total

    def print(self):
        accuracy = self.get()
        print(f"Accuracy: {accuracy*100:.1f}% "
              f"({self._num_correct}/{self._num_total})")
