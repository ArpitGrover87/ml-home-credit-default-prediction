import os


class FileUtils:

    @staticmethod
    def listdir(path):
        if os.path.exists(path):
            return os.listdir(path)
        else:
            return []

    @staticmethod
    def get_files_with_path(path):
        file_list = FileUtils.listdir(path)
        file_ref = dict()
        for f in file_list:
            file_ref[f] = path + '/' + f
        return file_ref
