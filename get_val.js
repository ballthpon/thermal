function get_class_score() {
  $.ajax({
    type: "GET",
    url: 'get_class_score',
    dataType: "text",
    contentType: 'application/text;charset=UTF-8',
    success: function (result) {
      $("#score_txt").html(result);
    }
  });
}

function get_time() {
  $.ajax({
    type: "GET",
    url: 'get_time',
    dataType: "text",
    contentType: 'application/text;charset=UTF-8',
    success: function (result) {
      $("#time_txt").html(result);
    }
  });
}

