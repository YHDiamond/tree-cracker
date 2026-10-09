"""Exercise the actual C++ runtime parser without requiring a CUDA device."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ObservationInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        directory = Path(cls.temp.name)
        source = directory / 'parse.cpp'
        source.write_text('''#include "src/Observation Input.cuh"
#include <iostream>
int main() {
  try {
    for (const auto &t : readTreeObservations(std::cin)) {
      std::cout << t.version << " " << t.type << " " << t.x << " " << t.z
                << " " << t.heightMin << " " << t.heightMax << " ";
      for (auto leaf : t.leaves) std::cout << leaf;
      std::cout << "\\n";
    }
  } catch (const std::exception &e) { std::cerr << e.what(); return 2; }
}''')
        cls.binary = directory / 'parse'
        subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                        '-I', str(ROOT), str(source), '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def parse(self, text):
        return subprocess.run([str(self.binary)], input=text, text=True, capture_output=True)

    def test_fresh_fixture_exact_data(self):
        result = self.parse((ROOT / 'Test Data/fresh-16-1-20261009.txt').read_text())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            '4 2 7 26 5 5 110010000011', '4 2 4 19 6 6 011011101010',
            '4 2 15 24 7 7 010101101110', '4 2 9 35 7 7 101110010000'])

    def test_unknown_attributes_and_negative_coordinates(self):
        result = self.parse('# comment\n\n1.16.1 Forest Oak -1 -16 0 0 ???????????? # unknown\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '4 0 -1 -16 -2147483648 2147483647 222222222222')

    def test_reject_malformed_or_unsupported_data(self):
        valid = '1.16.1 Forest Birch 7 26 5 5 110010000011'
        for text in ['', '# comments only', valid + ' seed=123',
                     valid.replace('1.16.1', '26.3'), valid.replace('Forest', 'Taiga'),
                     valid.replace('Birch', 'Spruce'), valid.replace('5 5', '4 5'),
                     valid.replace('5 5', '7 6'), valid.replace('7 26', '2147483648 26'),
                     valid.replace('110010000011', '11001000001'),
                     valid.replace('110010000011', '11001000001x'), valid + '\n' + valid,
                     valid + '\n1.16.1 Forest Unknown 7 26 0 0 ????????????',
                     '1.16.1 Forest Fancy_Oak 0 0 0 0 100000000000']:
            with self.subTest(text=text):
                result = self.parse(text)
                self.assertEqual(result.returncode, 2)
                self.assertTrue(result.stderr)

    def test_chunk_capacity_and_type_alternatives(self):
        rows = [f'1.16.1 Forest Birch {i} -1 0 0 ????????????' for i in range(16)]
        self.assertEqual(self.parse('\n'.join(rows)).returncode, 0)
        self.assertEqual(self.parse('\n'.join(rows + ['1.16.1 Forest Birch 0 -2 0 0 ????????????'])).returncode, 2)
        self.assertEqual(self.parse('\n'.join(rows + ['1.16.1 Forest Birch 16 -1 0 0 ????????????'])).returncode, 0)
        self.assertEqual(self.parse('\n'.join(rows + ['1.16.1 Forest Oak 0 -1 0 0 ????????????'])).returncode, 0)


if __name__ == '__main__':
    unittest.main()
