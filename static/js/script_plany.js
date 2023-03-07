$(document).ready(function () {
    $("[type='field']").click(function (event) {
        $("#" + event.target.id).attr("contenteditable", "true");
    });

    $("[type='field']").keydown(function (e) {
        if (e.which == 13) {
            $("#" + e.target.id).removeAttr("contenteditable");
        }
    });
});

const numberOfDaysInWeek = 7; // 7 or 5 only

class timeTable {
    constructor(htmlId, startDate, numberOfMonths, numberOfClassHours) {
        this.htmlId = htmlId;
        this.numberOfMonths = numberOfMonths;
        this.startDate = startDate;
        this.numberOfClassHours = numberOfClassHours;
    }


    drawTable() {
        $(this.htmlId).attr('style', 'table');
        $(this.htmlId).append(this.__drawAll().join(''));
    }

    __cloneDate(date) {
        return new Date(date.getTime());
    }

    __drawAll() {
        let data = this.__generateMonthsAndDays(this.startDate);
        let html = [];
        let monthBeforeAdded = false;
        //Header - months
        html.push('<tr class="dark-gray">');
        html.push('<td colspan="2">Mc-e</td>');
        for (let i = 0; i < this.numberOfMonths; i++) {
            if (i == 0) {
                if (this.startDate.getDay() != 1) {
                    let dt = this.__cloneDate(this.startDate);
                    dt.setDate(0);
                    html.push('<td colspan="1">' + this.__getRomanNum(dt.getMonth() + 1) + '</td>');

                }
            }
            html.push('<td colspan="' + data[i].numOfWeeks + '">' + data[i].roman + '</td>');
        }
        html.push('<td rowspan="2">Sale</td>');
        html.push('<td rowspan="2">Symbol</td>');
        html.push('<td rowspan="2">Przedmiot/Wykladowca</td>');
        html.push('<td rowspan="2">W</td>');
        html.push('<td rowspan="2">Cw</td>');
        html.push('<td rowspan="2">Spos. Zal.</td>');
        html.push('<td rowspan="2">ECTS</td>');
        html.push('</tr>');
        //EOHeader
        //Hours
        for (let dayOfWeek = 1; dayOfWeek <= numberOfDaysInWeek; dayOfWeek++) {
            let dayNo = dayOfWeek == 7 ? 0 : dayOfWeek;
            //Days
            html.push('<tr class="dark-gray">');
            html.push('<td colspan="2">dni</td>');
            for (let i = 0; i < this.numberOfMonths; i++) {
                for (let z = 0; z < data[i].days[dayNo].length; z++) {
                    html.push('<td>' + data[i].days[dayNo][z].getDate() + '</td>');
                }
            }
            html.push('</tr>');
            //EODays
            html.push('<tr><td rowspan="' + (this.numberOfClassHours + 1) + '"><div class="vertical-text">' + this.__getWeekDayName(dayNo) + '</div></td></tr>');
            for (let numOfClass = 1; numOfClass <= this.numberOfClassHours; numOfClass++) {
                html.push('<tr><td class="dark-gray">' + numOfClass + '</td>');
                for (let i = 0; i < data.length; i++) {
                    for (let weeksNum = 0; weeksNum < data[i].days[dayNo].length; weeksNum++) {
                        let day = data[i].days[dayNo][weeksNum];
                        let id = "class-" + numOfClass + "-" + data[i].year + "-" + (day.getMonth() + 1) + "-" + day.getDate();
                        html.push('<td id="' + id + '" type="field"></td>');
                    }
                }
                html.push('<td type="field" id="sala_' + dayOfWeek + '_' + numOfClass + '"></td>');
                html.push('<td type="field" id="symbol_' + dayOfWeek + '_' + numOfClass + '"></td>');
                html.push('<td type="field" id="wykladowca_' + dayOfWeek + '_' + numOfClass + '"></td>');
                html.push('<td type="field" id="wyklady_' + dayOfWeek + '_' + numOfClass + '"></td>');
                html.push('<td type="field" id="cwiczenia_' + dayOfWeek + '_' + numOfClass + '"></td>');
                html.push('<td type="field" id="sposzal_' + dayOfWeek + '_' + numOfClass + '"></td>');
                html.push('<td type="field" id="ects_' + dayOfWeek + '_' + numOfClass + '"></td>');
                html.push('</tr>');
            }
        }

        //EOHours
        return html;
    }

    __addMonth(date) {
        let day = date.getDate();
        date.setDate(15);
        date = new Date(new Date(date).setMonth(date.getMonth() + 1));
        date.setDate(day);
        return date;
    }

    __weeksinMonth(date) {
        let first = this.__cloneDate(date);
        let last = this.__cloneDate(date);
        first.setDate(1);
        last = this.__addMonth(last);
        last = new Date(last.getFullYear(), last.getMonth(), 0);
        let firstWeekNo = getWeekNumber(first);
        let lastWeekNo = getWeekNumber(last);
        if (firstWeekNo > lastWeekNo) {
            lastWeekNo += 52;
        }
        return lastWeekNo - firstWeekNo + 1;

    }

    __generateMonthsAndDays(date) {
        let dt = this.__cloneDate(date);
        dt.setDate(1);
        let data = [];
        let unshiftNeeded = false;
        for (let i = 0; i < this.numberOfMonths; i++) {
            let monthNo = dt.getMonth() + 1;
            let roman = this.__getRomanNum(monthNo);
            let numOfWeeks = this.__weeksinMonth(dt);
            if (dt.getDay() != 1) numOfWeeks--;
            let days = [];
            for (let day = 0; day <= 6; day++) {
                days[day] = this.__getSameDays(dt, day);
            }

            data[i] = {
                roman: roman,
                numOfWeeks: numOfWeeks,
                days: days,
                year: dt.getFullYear(),
                month: dt.getMonth() + 1
            };
            dt = this.__addMonth(dt);
        }
        if (this.startDate.getDay() != 1) {
            let firstMonday = this.__cloneDate(data[0].days[1][0]);
            let numOfDaysToUnshift = firstMonday.getDate() - 1;
            for (let i = 7; i > 0; i--) {
                if (i > numOfDaysToUnshift) {
                    let dayOfWeek = i == 1 ? 0 : 8 - i;
                    let dt = this.__cloneDate(data[0].days[dayOfWeek][0]);
                    dt.setDate(dt.getDate() - 7);
                    data[0].days[dayOfWeek].unshift(dt);
                }
            }
        }
        let lastMonday = this.__cloneDate(data[data.length - 1].days[1][data[data.length - 1].days[1].length - 1]);
        let lastDay = this.__cloneDate(lastMonday);
        lastDay.setDate(1);
        lastDay.setMonth(lastDay.getMonth() + 1);
        lastDay.setDate(0);
        let numOfDaysToPush = 6 - (daydiff(lastMonday, lastDay));
        let dat = this.__cloneDate(lastMonday);
        dat.setDate(1);
        dat.setMonth(dat.getMonth() + 1);
        for (let i = 8 - numOfDaysToPush; i <= 7; i++) {
            let dayNo = i == 7 ? 0 : i;
            data[data.length - 1].days[dayNo].push(this.__cloneDate(dat));
            dat.setDate(dat.getDate() + 1);
        }
        return data;
    }

    __getRomanNum(num) {
        while (num > 12) {
            num -= 12;
        }
        let nums = [
            '0', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'
        ];
        return nums[num];
    }

    __getWeekDayName(dayNo) {
        let names = [
            'Niedziela', 'Poniedzialek', 'Wtorek', 'Środa', 'Czwartek', 'Piątek', 'Sobota', 'Niedziela'
        ]
        return names[dayNo];
    }


    __getSameDays(date1, dayNumber) {
        let date = this.__cloneDate(date1);
        let month = date.getMonth(), sameDays = [];
        date.setDate(1);
        while (date.getDay() !== dayNumber) {
            date.setDate(date.getDate() + 1);
        }
        while (date.getMonth() === month) {
            sameDays.push(this.__cloneDate(date));
            date.setDate(date.getDate() + 7);
        }
        return sameDays;
    }

    __getFirstMondayDayNumber(date) {
        let dt = this.__cloneDate(date);
        dt.setDate(1);
        for (let i = 1; i < 10; i++) {
            if (dt.getDay() == 1) {
                console.log("s", dt);
                return dt.getDate();
            }
            dt.setDate(i);
        }

    }
}

function getWeekNumber(d) {
    d = new Date(+d);
    d.setHours(0, 0, 0, 0);
    d.setDate(d.getDate() + 4 - (d.getDay() || 7));
    let yearStart = new Date(d.getFullYear(), 0, 1);
    return Math.ceil(( ( (d - yearStart) / 86400000) + 1) / 7);
}

function daydiff(start, end) {
    return Math.abs(Math.floor(( start - end ) / 86400000));
}


