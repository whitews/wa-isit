app.controller(
    'MainController',
    [
        '$scope',
        'Image',
        'Region',
        function ($scope, Image, Region) {
            $scope.images = Image.query();
            $scope.current_image = null;
            $scope.init_regions = [];
            $scope.user_regions = null;

            $scope.image_selected = function (item) {
                $scope.current_image = item.id;
            };

            $scope.areas_changed = function (ev, boxId, areas, area) {
                $scope.user_regions = areas;
	        };

	        $scope.identify_regions = function () {
                $scope.user_regions.forEach(function(region) {
                    region_response = Region.get(
                        {
                            'id': $scope.current_image,
                            'x': region.x,
                            'y': region.y,
                            'w': region.width,
                            'h': region.height
                        }
                    );

                    region_response.$promise.then(function(data){
                        $scope.$broadcast("ngAreas:renameByAreaId", {
                            areaid : region.areaid,
                            name : data.predicted_class
                        });
                    });
                });
            };
        }
    ]
);
