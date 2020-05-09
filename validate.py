

def evaluateModelOnValidationSet(validation_aug = False):

	# Create a pytorch dataset
	data_dir = pathlib.Path('./data/tiny-imagenet-200')
	# image_count = len(list(data_dir.glob('**/*.JPEG')))
	CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
	# print('Discovered {} images'.format(image_count))

	assert(len(CLASS_NAMES) == 200)

	# Create the validation data generator
	batch_size = 1024
	im_height = 64
	im_width = 64

	# Should data augmentation be performed on the training data?
	if validation_aug == True:
	    validation_data_transforms = transforms.Compose([
	        transforms.ColorJitter(brightness = 1, contrast = 1, saturation = 1, hue = 1),
	        transforms.RandomAffine(degrees = 20, translate = 0.05, scale = None, shear = 10),
	        transforms.RandomGrayscale(p = 0.1),
	        transforms.RandomHorizontalFlip(p = 0.1),
	        transforms.RandomVerticalFlip(p = 0.1),
	        transforms.RandomRotation(degrees = 10),
	        transforms.RandomPerspective(p = 0.2),


	        transforms.ToTensor(),
	        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
	    ])
	elif validation_aug == False:
	    validation_data_transforms = transforms.Compose([
	        transforms.ToTensor(),
	        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
	    ])


	# Create the validation data generator
	validation_set = torchvision.datasets.ImageFolder(dataPathString + '/val/data', validation_data_transforms)
	validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size,
	                                       shuffle = True, num_workers = 1, pin_memory = True)

	# Size of Validation Data Set
	validationDataLength = len(validation_set)
	assert(validationDataLength == 10000)

	# Load Model and its Weights
	ckpt = torch.load('latest.pt')
	model = Net(len(CLASSES), im_height, im_width)
	model.load_state_dict(ckpt['net'])

	# Put the model in evaluation mode (to test on validation data)
	model.eval()


	# Loop through validation batches
	for idx, (inputs, targets) in enumerate(tqdm(validation_loader)):

	    inputs = inputs.to(device)
	    targets = targets.to(device)

	    # Run the model on the validation batch
	    outputs = model(inputs)

	    # Get validation loss and validation accuracy on this batch
	    loss = criterion(outputs, targets)
	    _, preds = torch.max(outputs, 1)

	    # Keep tracking of running statistics on validation loss and accuracy
	    running_loss += loss.item() * inputs.size(0)
	    running_corrects += torch.sum(preds == targets.data)

	validationLoss = running_loss / validationDataLength
	validationAccuracy = running_corrects.double() / validationDataLength

	return validationLoss, validationAccuracy


if __name__ == '__main__':
    validationLoss, validationAccuracy = evaluateModelOnValidationSet(validation_aug = False)
    print("validationLoss is: " + str(validationLoss))
    print("validationAccuracy is: " + str(validationAccuracy))

    

