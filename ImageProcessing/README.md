Steps to print:

1. python3 LoadTransfer ImageProcessing/LoadTransferLoaders.py on the relevant numpy arrays by editing models
2. python3 ExtraSpacePNG.py on the relevant produced directories.  Right now you have to run it once for each model/eps/label.  Will script this hopefully.
3. Print using print_all.sh in ImageProcessing
4. scanimage  -d 'fujitsu:fi-7600:1203571' --source='ADF Front' --format=tiff --mode=Color --resolution=200 --page-width=215.872 --page-height=279.4 -y 279.4  --batch-start=0 --batch='bubbles-%d.tiff' --prepick=on --buffermode=on
   -d <device>  (it includes the serial number)
   --source=‘ADF Front’  use the automatic document  feeder.
   --format=tiff : save as tiff
   --mode=Color  : We do RGB
   --resolution=200  : 200dpi
   --page-width and --page-height --> The values for US letter size. 279.4 millimeters is 11 inches.(215.872mm is 8.5 inches)
   -y 279.4 (the height of the scan to be made — in millimeters —. That’s of course the same as the page height).
   --batch-start=0  ---> to automatically number the files from 0
   --batch=‘bubbles-%d.tiff’   :  the template for the filename using the numbering from the batch-start where %d is.
   --prepick and --buffermode are device specific.Scan using whatever
5. Register using ImageRegistration.py
6. Get bubbles back with ExtractBubblesFromWhitespacePNG.py
7. Return to a numpy array with LoadScannedBubbles.py
