#include <random>
#include <inttypes.h>

#include "toolchain.h"
#include "toolkit_filter.h"

cifar::CIFAR10_dataset<std::vector, std::vector<double>, uint8_t> Toolchain_denoise::dataset(cifar::read_dataset<std::vector, std::vector, double, uint8_t>(CIFAR_10_DATA_LOCATION));
cifar::CIFAR10_dataset<std::vector, std::vector<double>, uint8_t> Toolchain_denoise::noisy_dataset(cifar::read_dataset<std::vector, std::vector, double, uint8_t>(NOISY_CIFAR_10_DATA_LOCATION));
const uint8_t max_filtering = 3;

void Toolchain_denoise::getData_Image(std::vector<double>& img) const{
    for (int index = 0; index < 32*32*3; ++index) {
        img[index] = ((double) *((this->currentImage.getDataAt(typeid(double), index)).getSharedPointer<const double>()));
    }
}

void Toolchain_denoise::setData_Image(std::vector<double> &img) {
    for (int index = 0; index < 32*32*3; ++index) {
        this->currentImage.setDataAt(typeid(double), index, img.at(index));
    }
}

void Toolchain_denoise::changeCurrentImage()
{
	// Get the container for the current mode.
	std::vector<std::vector<double>>& dataSource = (this->currentMode == Learn::LearningMode::TRAINING) ?
		this->noisy_dataset.training_images : this->noisy_dataset.test_images;

	// Select the image 
	// If the mode is training or validation
	if (this->currentMode == Learn::LearningMode::TRAINING || this->currentMode == Learn::LearningMode::VALIDATION) {
		// Select a random image index
		this->currentIndex = rng.getUnsignedInt64(0, dataSource.size() - 1);
	}
	else {
        //Save the image
        if(this->currentIndex >= 0){
            vector<double> img_saved(32*32*3);
            Toolchain_denoise::getData_Image(img_saved);
            toolkit::save_Image(img_saved, this->currentIndex);
        }
		// If the mode is TESTING, just go to the next image
		this->currentIndex = (this->currentIndex + 1) % dataSource.size();
	}

	// Load the image in the dataSource
	for (uint64_t pxlIndex = 0; pxlIndex < 32 * 32 * 3; pxlIndex++) {
		this->currentImage.setDataAt(typeid(double), pxlIndex, dataSource.at(this->currentIndex).at(pxlIndex));
	}
    // Update previous image with the current one
    this->img_before = dataSource[this->currentIndex];

	//Compute MSE, normally they are equals
    this->MSE_after = this->MSE_before = toolkit::MSE_compute(dataSource[currentIndex], (this->img_before));

}

Toolchain_denoise::Toolchain_denoise() : LearningEnvironment(7), currentImage(32 * 3, 32 ),img_before(32*32*3),nb_action_filtering(0),global_reward(0)
{
	// Fill shared dataset dataset(mnist::read_dataset<std::vector, std::vector, double, uint8_t>(MNIST_DATA_LOCATION))
	if (Toolchain_denoise::dataset.training_labels.size() != 0) {
		std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
		std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
		std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
		std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
	}
    if (Toolchain_denoise::noisy_dataset.training_labels.size() != 0) {
        std::cout << "Nbr of noisy training images = " << noisy_dataset.training_images.size() << std::endl;
        std::cout << "Nbr of noisy training labels = " << noisy_dataset.training_labels.size() << std::endl;
        std::cout << "Nbr of noisy test images = " << noisy_dataset.test_images.size() << std::endl;
        std::cout << "Nbr of noisy test labels = " << noisy_dataset.test_labels.size() << std::endl;
    }

	else {
		throw std::runtime_error("Initialization of CIFAR-10 databased failed.");
	}
}

void Toolchain_denoise::doAction(uint64_t actionID)
{
    //If the number of filter applied on the current exceed the max value of 3
    if(this->nb_action_filtering >= max_filtering){
        this->changeCurrentImage();
        this->nb_action_filtering = 0;
    }
    else {
        vector<double> img(32 * 32 * 3);
        Toolchain_denoise::getData_Image(img);
        this->img_before = img;//copy current img before filtering process
        this->nb_action_filtering ++; //Increase the number of applied filter on one image

        //Select the filter
        switch (actionID) {
            case 0:
                toolkit::mean_filter_3x3(img, img);
                break;
            case 1:
                toolkit::mean_filter_5x5(img, img);
                break;
            case 2:
                toolkit::median_filter_3x3(img, img);
                break;
            case 3:
                toolkit::median_filter_5x5(img, img);
                break;
            case 4:
                toolkit::gaussian_filter_3x3(img, img);
                break;
            case 5:
                toolkit::gaussian_filter_5x5(img, img);
                break;
            case 6:
                this->changeCurrentImage();
                this->nb_action_filtering = 0;
                break;
            default:
                break;
        }
        //If the action wasn't changeCurrentImage, save the new image and compute score
        if (actionID >= 0 && actionID < 6){
            Toolchain_denoise::setData_Image(img);
            Toolchain_denoise::compute_score_filter(img);
        }
    }
}

void Toolchain_denoise::compute_score_filter(const std::vector<double> & img_after) {
    //Compute the reward
    // Get the container for the current mode.
    std::vector<std::vector<double>>& dataSource = (this->currentMode == Learn::LearningMode::TRAINING) ?
                                                   this->dataset.training_images : this->dataset.test_images;


    //Compute MSE of the current image after the action and the image before
    this->MSE_after = toolkit::MSE_compute(dataSource[currentIndex],img_after);
    this->MSE_before = toolkit::MSE_compute(dataSource[currentIndex], this->img_before);

    this->global_reward += (this->MSE_before - this->MSE_after)/1000;

    //Normalization (not a good idea because of smoothing filters which decreasing slowly the MSE and obtained a better global reward
    /*
    if (this->MSE_before > this->MSE_after){ //good
        this->global_reward+= 1;
    }
    else if(this->MSE_before < this->MSE_after){//bad
        this->global_reward+=  -0.25;
    }*/
}

void Toolchain_denoise::reset(size_t seed, Learn::LearningMode mode)
{
    // Create seed from seed and mode
    size_t hash_seed = Data::Hash<size_t>()(seed) ^Data::Hash<Learn::LearningMode>()(mode);
    this->rng.setSeed(hash_seed);
	this->currentMode = mode;

	this->nb_action_filtering=0;
	this->global_reward=0;
	// Reset at -1 so that in TESTING mode, first value tested is 0.
	this->currentIndex = -1;
	this->changeCurrentImage();
}

std::vector<std::reference_wrapper<const Data::DataHandler>> Toolchain_denoise::getDataSources()
{
	std::vector<std::reference_wrapper<const Data::DataHandler>> res = { currentImage };

	return res;
}

bool Toolchain_denoise::isCopyable() const
{
	return true;
}

Learn::LearningEnvironment* Toolchain_denoise::clone() const
{
	return new Toolchain_denoise(*this);
}

double Toolchain_denoise::getScore() const
{
    return this->global_reward;
}

bool Toolchain_denoise::isTerminal() const
{
	return false;
}

void Toolchain_denoise::denoise_data_test(const Environment& env, const TPG::TPGVertex* bestRoot) {
	// Printing in a file results of the best root
    ofstream wf;
    cout << "Print best root results in a file" << endl;
    cout << "Saving in dat folder filtered images" << endl;
    wf.open("Action_taken.txt",ios::out);
    if(!wf) {
        cout << "Cannot open file !" << endl;
    }
    String info;

    TPG::TPGExecutionEngine tee(env, NULL);

	// Change the MODE of CIFAR-10
	this->reset(0, Learn::LearningMode::TESTING);

    auto index = this->currentIndex;
	const int TOTAL_NB_IMAGE = 10000;
    for (u_int64_t i =0; i < 5 * TOTAL_NB_IMAGE && index < TOTAL_NB_IMAGE; i++) {

		// Execute
		auto path = tee.executeFromRoot(*bestRoot);
		const TPG::TPGAction* action = (const TPG::TPGAction*)path.at(path.size() - 1);
		uint8_t actionID = (uint8_t)action->getActionID();

        // Do action (to trigger image update)
        this->doAction(action->getActionID());

        if (index != this->currentIndex){
            wf << "------------ " << endl;

            index++;
        }

		info = "Index :" + to_string(this->currentIndex) + ", ActionID :" + to_string(actionID) + ", MSE_before :" + to_string(this->MSE_before) + ", MSE_after :" + to_string(this->MSE_after);
        wf << info << endl;

	}
    wf.close();
}
