import unittest
import os
import tempfile


example_param_path = '/pscratch/sd/j/joeschm/NSXTU_discharges/129015/r_0.870001_PE_top/kymin_scan/scanfiles0007/parameters_0001'
example_field_path = '/pscratch/sd/j/joeschm/NSXTU_discharges/129015/r_0.870001_PE_top/kymin_scan/scanfiles0007/field_0001'


from TPED.src.utils.file_functions import suffix_from_filename, switch_suffix_file, FileError

class TestSuffixFromFilename(unittest.TestCase):
    def test_suffix_from_filename(self):
        self.assertEqual(suffix_from_filename('omega_0032'), '0032')
        self.assertEqual(suffix_from_filename('parameters'), '.dat')
        
        with self.assertRaises(ValueError):
            suffix_from_filename('nrg_123')

        with self.assertRaises(ValueError):
            suffix_from_filename('field_12345')





class TestSwitchSuffixFIle(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for testing
        self.test_file = tempfile.NamedTemporaryFile(delete=False)
        self.test_file.close()

    def tearDown(self):
        # Remove the temporary file after testing
        os.unlink(self.test_file.name)

    def test_switch_suffix_file(self):

        # Test case where the file does not exist
        with self.assertRaises(FileError):
            switch_suffix_file('nonexistent_file', 'new')

        # Test case where the file is not a file
        with self.assertRaises(FileError):
            switch_suffix_file('/', 'new')
        

        
        # Test case where the filename 'parameters_0002' changes to 'field_0002'
        with self.assertEqual(example_param_path, example_field_path):
            switch_suffix_file(example_param_path, example_field_path)
        




if __name__ == '__main__':
    unittest.main()









