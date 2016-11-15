// Reference: https://github.com/rstaib/jquery-steps/wiki (lots of options to customize this form)
$(document).ready(function () {

    //Small Event that will resize the step content when the window width change
    window.addEventListener('resize', function (event) {
        $("#form .body.current").each(function () {
            var bodyHeight = $(this).height();
            var padding = $(this).innerHeight() - bodyHeight;
            bodyHeight += padding;
            $(this).parent().css("height", bodyHeight + "px");
        });
    });


    // $("#wizard").steps();
    // $("#form").steps({
    //     bodyTag: "section",
    //     showFinishButtonAlways: false,
    //     onStepChanging: function (event, currentIndex, newIndex) {
    //         // Always allow going backward even if the current step contains invalid fields!
    //         if (currentIndex > newIndex) {
    //             return true;
    //         }
    //
    //         //Fills in information for second step based on dataset choice
    //         if (currentIndex === 2) {
    //             $("#form").steps("remove", 3);
    //             $("#form").steps("insert", 3, {
    //                 title: "Parameters",
    //                 contentMode: "async",
    //                 contentUrl: "/analysis/getmodelform?modelid=" + $("#model").val() + "&datasetid=" + $("#dataset").val()
    //             });
    //         }
    //
    //         var form = $(this);
    //
    //         // Clean up if user went backward before
    //         if (currentIndex < newIndex) {
    //             // To remove error styles
    //             $(".body:eq(" + newIndex + ") label.error", form).remove();
    //             $(".body:eq(" + newIndex + ") .error", form).removeClass("error");
    //         }
    //
    //         // Disable validation on fields that are disabled or hidden.
    //         form.validate().settings.ignore = ":disabled,:hidden";
    //
    //         // Start validation; Prevent going forward if false
    //         return form.valid();
    //     },
    //     onFinishing: function (event, currentIndex) {
    //         var form = $(this);
    //
    //         // Disable validation on fields that are disabled.
    //         // At this point it's recommended to do an overall check (mean ignoring only disabled fields)
    //         form.validate().settings.ignore = ":disabled";
    //
    //         // Start validation; Prevent form submission if false
    //         return form.valid();
    //     },
    //     onFinished: function (event, currentIndex) {
    //         var form = $(this);
    //
    //         submitForm(form.serialize());
    //     }
    // }).validate({
    //     errorPlacement: function (error, element) {
    //         element.before(error);
    //     }
    // });

    $("#model").change(function () {
        $("#model-description-" + this.value).show().siblings().hide();
    });

    $("#model").change();

    ///////////////////////////////////
    // Pre-processing Tab
    ///////////////////////////////////

    $('.i-checks').iCheck({
        checkboxClass: 'icheckbox_square-green',
        radioClass: 'iradio_square-green'
    });

    $('#undersampling_rate-slider').noUiSlider({
        start: [50],
        step: 1,
        connect: "lower",
        range: {
            'min': [1],
            'max': [100]
        }
    });
    $('#undersampling_rate-slider').Link('lower').to($('#undersampling_rate-value'));
    $('#undersampling_rate-slider').Link('lower').to($('#undersampling_rate'));
    $('#undersampling_rate-slider').attr('disabled', 'disabled');


    $('#oversampling_percentage-slider').noUiSlider({
        start: [100],
        step: 10,
        connect: "lower",
        range: {
            'min': [10],
            'max': [1000]
        }
    });
    $('#oversampling_percentage-slider').Link('lower').to($('#oversampling_percentage-value'));
    $('#oversampling_percentage-slider').Link('lower').to($('#oversampling_percentage'));
    $('#oversampling_percentage-slider').attr('disabled', 'disabled');

    $('#pca-slider').noUiSlider({
        start: [5],
        step: 1,
        connect: "lower",
        range: {
            'min': [1],
            'max': [10] // TODO: Replace this by NUM_FEATUERS
        }
    });
    $('#pca-slider').Link('lower').to($('#n_components-value'));
    $('#pca-slider').Link('lower').to($('#n_components'));
    $('#pca-slider').attr('disabled', 'disabled');

    $('#binarization-slider').noUiSlider({
        start: [0],
        step: 0.1,
        connect: "lower",
        range: {
            'min': [-1], // TODO: Replace this by -MAX(featurevalue)
            'max': [1] // TODO: Replace this by MAX(featurevalue)
        }
    });
    $('#binarization-slider').Link('lower').to($('#binarization_threshold-value'));
    $('#binarization-slider').Link('lower').to($('#binarization_threshold'));
    $('#binarization-slider').attr('disabled', 'disabled');


    $("#q1").qtip({
        content: {
            text: 'Adjust class distribution by removing instances from the majority class. Select the desired amount of majority class instances to be used for training .',
            title: '<i>Undersampling<\/i>'
        },
        style: {classes: 'qtip-rounded qtip-shadow qtip-cream'}
    });
    $("#q2").qtip({
        content: {
            text: 'Use SMOTE to create synthetic instances of the minority class. Use the slider to select the percentage of new instances to create.',
            title: '<i>Oversampling <\/i>'
        },
        style: {classes: 'qtip-rounded qtip-shadow qtip-cream'}
    });
    $("#q3").qtip({
        content: {
            text: 'Select the strategy to be used for missing value imputation.',
            title: '<i>Handing of Missing Data <\/i>'
        },
        style: {classes: 'qtip-rounded qtip-shadow qtip-cream'}
    });
    $("#q4").qtip({
        content: {
            text: 'Linear dimensionality reduction using Singular Value Decomposition of the data and keeping only the most significant singular vectors to project the data to a lower dimensional space.',
            title: '<i>Principal Component Analysis <\/i>'
        },
        style: {classes: 'qtip-rounded qtip-shadow qtip-cream'}
    });
    $("#q5").qtip({
        content: {
            text: 'Standardize features by removing the mean and scaling to unit variance.',
            title: '<i>Standardization <\/i>'
        },
        style: {classes: 'qtip-rounded qtip-shadow qtip-cream'}
    });
    $("#q6").qtip({
        content: {
            text: 'Scale input vectors individually to unit norm (vector length).',
            title: '<i>Normalization <\/i>'
        },
        style: {classes: 'qtip-rounded qtip-shadow qtip-cream'}
    });
    $("#q7").qtip({
        content: {
            text: 'Binarize data (set feature values to 0 or 1) according to a threshold. Values greater than the threshold map to 1, while values less than or equal to the threshold map to 0. With the default threshold of 0, only positive values map to 1.',
            title: '<i>Binarization <\/i>'
        },
        style: {classes: 'qtip-rounded qtip-shadow qtip-cream'}
    });
    $("#q8").qtip({
        content: {
            text: 'Remove rows that contain column values that are further than 3 standard deviations from mean.',
            title: '<i>Outlier Detection <\/i>'
        },
        style: {classes: 'qtip-rounded qtip-shadow qtip-cream'}
    });

    $('.i-checks').on('ifChecked', function () {
        var parameter = $(this).find(":checked").attr('name');
        var value = $(this).find(":checked").val();
        if (parameter === 'undersampling') {
            if (value === 0) {
                $('#undersampling_rate-slider').attr('disabled', 'disabled');
                $('#undersampling_rate-value').attr('class', "parameter_value_off");
                $('#undersampling_rate-value2').attr('class', "parameter_value_off");
            }
            else {
                $('#undersampling_rate-slider').removeAttr('disabled');
                $('#undersampling_rate-value').attr('class', "parameter_value");
                $('#undersampling_rate-value2').attr('class', "parameter_value");
            }
        }
        else if (parameter === 'oversampling') {
            if (value === 0) {
                $('#oversampling_percentage-slider').attr('disabled', 'disabled');
                $('#oversampling_percentage-value').attr('class', "parameter_value_off");
                $('#oversampling_percentage-value2').attr('class', "parameter_value_off");
            }
            else {
                $('#oversampling_percentage-slider').removeAttr('disabled');
                $('#oversampling_percentage-value').attr('class', "parameter_value");
                $('#oversampling_percentage-value2').attr('class', "parameter_value");
            }
        }
        else if (parameter === 'pca') {
            if (value === 0) {
                $('#pca-slider').attr('disabled', 'disabled');
                $('#n_components-value').attr('class', "parameter_value_off");
            }
            else {
                $('#pca-slider').removeAttr('disabled');
                $('#n_components-value').attr('class', "parameter_value");
            }
        }
        else if (parameter === 'normalization') {
            if (value === 0) {
                $('#norm').prop('disabled', 'disabled');
            }
            else {
                $('#norm').prop('disabled', false);
            }
        }
        else if (parameter === 'binarization') {
            if (value === 0) {
                $('#binarization-slider').attr('disabled', 'disabled');
                $('#binarization_threshold-value').attr('class', "parameter_value_off");
            }
            else {
                $('#binarization-slider').removeAttr('disabled');
                $('#binarization_threshold-value').attr('class', "parameter_value");
            }
        }
        else if (parameter === 'outlier_detection') {
            if (value === 0) {
                $('#outlier_detection_method').prop('disabled', 'disabled');
            }
            else {
                $('#outlier_detection_method').prop('disabled', false);
            }
        }
    });

});

function submitForm(params) {
    window.location.href = '/analysis/run?' + params;
}
