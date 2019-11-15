$(document).on('click', '[data-toggle="lightbox"]', function (event) {
    event.preventDefault();
    $(this).ekkoLightbox({
        alwaysShowClose: true
    });
});

$('#theme-switch').on('click', function () {
    var newTheme = $('body').hasClass('dark-theme') ? 'light' : 'dark';
    setTheme(newTheme);
});

$(document).ready(function () {
    $('#theme-switch').prop('disabled', false);
    var wantDark = getURLParameter('dark');
    var wantLight = getURLParameter('light');

    if (wantDark === true) {
        setTheme('dark');
    } else if (wantLight === true) {
        setTheme('light');
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        // don't modify urls or history when following media query preference
        setTheme('dark', modLinks=false);
    } else {
        setTheme('light', modLinks=false);
    }
});

function setTheme(theme, modLinks=true) {
    if (theme === 'dark') {
        $('body').addClass('dark-theme');
    } else {
        $('body').removeClass('dark-theme');
    }

    var otherTheme = theme === 'dark' ? 'light' : 'dark';
    $('#theme-switch span').text(otherTheme + ' theme');

    if (modLinks === true) {
        $('.local-link').each(function() {
            var href = $(this)[0].href.split('?', 1)[0];
            $(this)[0].href = href + '?' + theme;
        });

        history.replaceState({theme: true}, document.title, '?' + theme);
    }
}

function getURLParameter(param) {
    var params = window.location.search.substring(1).split('&');
    for (var i = 0; i < params.length; i++) {
        var param_val_i = params[i].split('=');
        if (param_val_i[0] === param) {
            if (param_val_i[1] === undefined) {
                return true;
            } else {
                return decodeURIComponent(param_val_i[1]);
            }
        }
    }
};
