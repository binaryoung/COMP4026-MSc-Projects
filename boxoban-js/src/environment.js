class Environment{
    /*
    wall = 0
    empty = 1
    target = 2
    box = 3
    box on target = 4
    player = 5
    player on target = 6

    push up = 0
    push down = 1
    push left = 2
    push right = 3
    move up = 4   
    move down = 5
    move left = 6
    move right = 7
    */

    moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    height = 10
    width = 10
    total_boxes = 4

    room = null
    topology = null
    player_location = null
    boxes_on_target = 0

    steps = 0
    max_steps = 128

    reward_per_step = -0.01
    reward_box_on_target = 1
    reward_box_off_target = -1
    reward_finished = 10

    constructor(room, topology){
        this.room = room
        this.topology = topology

        for (const [i, row] of room.entries()) {
            for (const [j, state] of row.entries()) {
              if (state == 5) {
                this.player_location = [i, j]
              }
            }
        }
    }

    step(action){
        this.steps += 1
        let reward = this.reward_per_step

        if (action <= 3){
            reward += this.push(action)
        } else {
            reward += this.move(action)
        }

        let finished = (this.boxes_on_target == this.total_boxes)
        let max_steps = (this.steps >= this.max_steps)
        let done = finished || max_steps

        if (finished == true){
            reward += this.reward_finished
        }

        let info = {
            "finished": finished,
            "max_steps": max_steps
        }

        return {
            step: this.steps,
            reward,done,info
        }
    }

    push(action){
        let change = this.moves[action % 4]

        let player_location = JSON.parse(JSON.stringify(this.player_location))
        let next_player_location = [player_location[0] + change[0], player_location[1] + change[1]]
        let next_box_location = [next_player_location[0] + change[0], next_player_location[1] + change[1]]

        if (next_box_location[0] >= this.height || next_box_location[1] >= this.width){
            return 0
        }

        let next_player_location_state = this.room[next_player_location[0]][next_player_location[1]]
        let next_box_location_state = this.room[next_box_location[0]][next_box_location[1]]

        if (
            !([3, 4].includes(next_player_location_state) &&
            [1, 2].includes(next_box_location_state))
        ){
            return this.move(action)
        }

        let reward = 0

        //Move Player
        this.player_location = next_player_location
        this.room[next_player_location[0]][next_player_location[1]] = next_player_location_state == 4  ? 6 : 5
        this.room[player_location[0]][player_location[1]] = this.topology[player_location[0]][player_location[1]]

        if (next_player_location_state == 4){
            this.boxes_on_target -= 1
            reward += this.reward_box_off_target
        }

        //Move Box
        if (next_box_location_state == 2){
            this.room[next_box_location[0]][next_box_location[1]] = 4
            this.boxes_on_target += 1
            reward += this.reward_box_on_target
        } else {
            this.room[next_box_location[0]][next_box_location[1]] = 3
        }
        
        return reward
    }

    move(action){
        let change = this.moves[action% 4]

        let player_location = JSON.parse(JSON.stringify(this.player_location))
        let next_player_location = [player_location[0] + change[0], player_location[1] + change[1]]
        
        let next_player_location_state = this.room[next_player_location[0]][next_player_location[1]] 

        if (![1, 2].includes(next_player_location_state)){
            return 0
        }
        
        this.player_location = next_player_location

        this.room[next_player_location[0]][next_player_location[1]] = next_player_location_state == 2 ? 6 : 5
        this.room[player_location[0]][player_location[1]] = this.topology[player_location[0]][player_location[1]]

        return 0
    }
}

export { Environment as default }