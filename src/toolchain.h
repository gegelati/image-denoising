#ifndef TOOLCHAIN_H
#define TOOLCHAIN_H

#include <random>

#include <gegelati.h>
#include <iostream>

#include "cifar/cifar10_reader.hpp"

/**
* \brief LearningEnvironment to train an agent to denoise the CIFAR-10 database.
*/
class Toolchain_denoise : public Learn::LearningEnvironment {
protected:
	/// CIFAR-10 dataset for the training.
	static cifar::CIFAR10_dataset<std::vector, std::vector<double>, uint8_t> dataset;
	static cifar::CIFAR10_dataset<std::vector, std::vector<double>, uint8_t> noisy_dataset;

	/// Current LearningMode of the LearningEnvironment.
	Learn::LearningMode currentMode;

	/// Randomness control
	Mutator::RNG rng;

	/// Current image provided to the LearningAgent
	Data::PrimitiveTypeArray2D<double> currentImage;

	/// Previous image before the filtering
    std::vector<double> img_before;

	/// Current index of the image in the dataset.
	uint64_t currentIndex;

	/// Counting the number of filters applied to the current image
	uint8_t nb_action_filtering;

	/// Global Reward
	double global_reward;

	/// MSE before the filtering
	double MSE_before;

	///MSE after the filtering
	double MSE_after;

	/**
	* \brief Change the image currently available in the dataSources of the LearningEnvironment.
	*
	* A random image from the dataset for the current mode is selected.
	*/
	void changeCurrentImage();

    /**
    * \brief Get the data from the image currently selected
    */
	void getData_Image(std::vector<double>& img) const;

    /**
    * \brief Set the data from the image currently selected
    */
    void setData_Image(std::vector<double>& img);

    /**
     * \brief Compute the score of the previous filtering
     */
    void compute_score_filter(const std::vector<double> &img_after);

public:

	/**
	* \brief Constructor.
	*
	* Loads the dataset on construction. Sets the LearningMode to TRAINING.
	* Sets the score to 0.
	*/
    Toolchain_denoise();

	/// Inherited via LearningEnvironment
	virtual void doAction(uint64_t actionID) override;

	/// Inherited via LearningEnvironment
	virtual void reset(size_t seed = 0, Learn::LearningMode mode = Learn::LearningMode::TRAINING) override;

	/// Inherited via LearningEnvironment
	virtual std::vector<std::reference_wrapper<const Data::DataHandler>> getDataSources() override;

	/// Inherited via LearningEnvironment
	virtual bool isCopyable() const override;

	/// Inherited via LearningEnvironment
	virtual LearningEnvironment* clone() const;

	/**
	* \brief Get the score of the current evaluation session (i.e. since the last reset).
	*
	* For the CIFAR-10 LearningEnvironment, the score is computed as follows:
	* - Score is incremented or decremented by the calculation of Score += MSE_before - MSE_after
	* (MSE_before -> before the filtering, MSE_after -> after the filtering)
	* - Score is left unchanged for action ID 0 which corresponds to an
	*   change of image or if the number max of filtering is reach.
	*
	* To be used in a viable learning process, score can be compared only on
	* agent evaluated on the same number of samples.
	*/
	virtual double getScore() const override;

	/**
	* \brief This LearningEnvironment will never reach a
	* terminal state.
	*
	* \return false.
	*/
	virtual bool isTerminal() const override;

	/**
	* \brief Function filtering the CIFAR-10's TESTING dataset and printing/saving the result
	*
	* \param[in] result the Map containing the list of roots within a TPGGraph,
	* with their score in ascending order.
	*/
	void denoise_data_test(const Environment& env, const TPG::TPGVertex* bestRoot);
};

#endif
