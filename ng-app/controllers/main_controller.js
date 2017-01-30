app.controller(
    'MainController',
    [
        '$scope',
        '$timeout',
        'Image',
        'Region',
        function ($scope, $timeout, Image, Region) {
            $scope.images = Image.query();
            $scope.current_image = null;
            $scope.init_regions = [];
            $scope.user_regions = null;
            $scope.region_probs = {};
            $scope.selected_region_probs = [];

            $scope.image_selected = function (item) {
                $scope.$broadcast('ngAreas:remove_all', {});
                $scope.current_image = item.id;
                $scope.current_image_name = item.name;
            };

            $scope.areas_changed = function (ev, area, areas) {
                $scope.user_regions = areas;
                $scope.selected_region_probs = null;

                if (area.z > 0) {
                    $scope.selected_region_probs = $scope.region_probs[area.areaid];

                    // since areas_changed is called from within the ngAreas directive, sometimes the
                    // new $scope doesn't get applied to the partial, but a generic $scope.$apply will
                    // cause other issues, so call apply at the end of the call stack
                    $timeout(function() {
                        $scope.$apply();
                    }, 0);
                }
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
                        $scope.region_probs[region.areaid] = data.probabilities;

                        if (region.z > 0) {
                            $scope.selected_region_probs = data.probabilities;
                        }
                    });
                });
            };

            $scope.delete_all_regions = function () {
                $scope.$broadcast("ngAreas:remove_all");
            };
        }
    ]
);
