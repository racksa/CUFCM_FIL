import func
import sys

driver = func.DRIVER()





if(len(sys.argv) > 1):
    if 'HPC' in sys.argv:
        driver.run_on_hpc = True

    if(sys.argv[1] == 'clean'):
        driver.delete_files()

    if(sys.argv[1] == 'view'):
        driver.view_files()

    if(sys.argv[1] == 'create_rules'):
        driver.create_rules()

    if(sys.argv[1] == 'run'):
        if(len(sys.argv) > 3):
            driver.current_thread = int(sys.argv[2])
            driver.num_thread = int(sys.argv[3])
        if(len(sys.argv) > 4):
            driver.cuda_device = int(sys.argv[4])
        if(len(sys.argv) > 5):
            driver.update_date(sys.argv[5])
        driver.create_rules()
        driver.run()

    if(sys.argv[1] == 'run_rules'):
        if(len(sys.argv) > 4):
            driver.current_thread = int(sys.argv[2])
            driver.num_thread = int(sys.argv[3])
            driver.cuda_device = int(sys.argv[4])
        driver.run()



























#